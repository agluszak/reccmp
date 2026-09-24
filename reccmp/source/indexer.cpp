// Emit the source-index records for one translation unit directly from Clang's
// AST. Partitioning by link namespace / TU, winner selection, conflicts, marker
// binding, asserted sizes and vtable addresses stay in reccmp, which owns them.
//
// Single-TU mode writes NDJSON to stdout. Batch mode (`--batch <manifest.jsonl>`)
// indexes many units in one process so LLVM target initialization happens once.
//
// Output is one JSON object per line: `{"record":"declaration",...}`,
// `{"record":"variable",...}`, `{"record":"class",...}`,
// `{"record":"member-use",...}`, `{"record":"marker-block",...}`,
// `{"record":"size-assertion",...}`, `{"record":"unit-abi",...}` or
// `{"record":"dependency",...}`.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

namespace {

using namespace clang;

// The collector supplies the physical compilation root, with a trailing slash.
std::string repositoryPrefix;
llvm::StringRef kRepositoryPrefix;

bool inRepository(llvm::StringRef path) { return path.starts_with(kRepositoryPrefix); }

std::string relative(llvm::StringRef path) {
  return inRepository(path) ? path.drop_front(kRepositoryPrefix.size()).str() : path.str();
}

std::string qualify(llvm::StringRef scope, llvm::StringRef name) {
  return scope.empty() ? name.str() : (scope + "::" + name).str();
}

std::string join(const std::vector<std::string>& parts, llvm::StringRef separator) {
  std::string result;
  for (size_t index = 0; index < parts.size(); ++index) {
    if (index) result += separator;
    result += parts[index];
  }
  return result;
}

struct Location {
  std::string file;
  unsigned line = 0;
  unsigned endLine = 0;
  unsigned column = 0;
  int64_t offset = -1;
};

struct CachedFile {
  bool indexed = false;
  std::string absolute;
};

using Clock = std::chrono::steady_clock;

double millisecondsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Where one translation unit's indexing time went, and what it produced.
// Emitted as the unit's last record so reccmp can report it; it is never
// part of the index itself.
struct Profile {
  double invocationMs = 0;   // driver, -cc1 job and CompilerInvocation
  double frontendMs = 0;     // ExecuteAction: parse, Sema and our consumer
  double consumerMs = 0;     // HandleTranslationUnit, inside frontendMs
  double memberUseMs = 0;    // member-use traversal, inside consumerMs
  double markerMs = 0;       // marker blocks, inside consumerMs
  double serializeMs = 0;    // JSON rendering and writing, inside consumerMs
  llvm::StringMap<int64_t> records;
  llvm::StringMap<int64_t> bytes;

  llvm::json::Object toJson() const {
    llvm::json::Object counts, sizes;
    for (const auto& entry : records) counts[entry.getKey()] = entry.getValue();
    for (const auto& entry : bytes) sizes[entry.getKey()] = entry.getValue();
    return llvm::json::Object{
        {"record", "profile"},
        {"invocation_ms", invocationMs},
        {"frontend_ms", frontendMs},
        {"consumer_ms", consumerMs},
        {"member_use_ms", memberUseMs},
        {"marker_ms", markerMs},
        {"serialize_ms", serializeMs},
        {"records", std::move(counts)},
        {"bytes", std::move(sizes)},
    };
  }
};

// Accumulates the time of one scope into a profile field.
class ScopedTimer {
 public:
  explicit ScopedTimer(double& total) : total_(total), start_(Clock::now()) {}
  ~ScopedTimer() { total_ += millisecondsSince(start_); }

 private:
  double& total_;
  Clock::time_point start_;
};

// One `//` comment that is the first thing on its line: the only shape a
// reccmp marker (or the name line completing one) can take.
struct LineComment {
  unsigned offset = 0;
  unsigned endOffset = 0;
  unsigned line = 0;
  unsigned column = 0;
  std::string text;
};

// The preprocessor reports every comment it lexes, which is exactly the set of
// comments in active code: markers inside `#if 0` never reach the index.
class LineCommentCollector : public CommentHandler {
 public:
  bool HandleComment(Preprocessor& preprocessor, SourceRange range) override {
    SourceManager& sources = preprocessor.getSourceManager();
    SourceLocation begin = range.getBegin();
    if (!begin.isFileID()) return false;
    auto [file, offset] = sources.getDecomposedLoc(begin);
    bool invalid = false;
    llvm::StringRef buffer = sources.getBufferData(file, &invalid);
    if (invalid || !buffer.substr(offset).starts_with("//")) return false;
    unsigned lineStart = offset;
    while (lineStart > 0 && buffer[lineStart - 1] != '\n' && buffer[lineStart - 1] != '\r') {
      --lineStart;
    }
    if (!buffer.slice(lineStart, offset).trim(" \t\f\v").empty()) return false;
    unsigned endOffset = sources.getFileOffset(range.getEnd());
    comments_[file].push_back(LineComment{
        offset,
        endOffset,
        sources.getLineNumber(file, offset),
        offset - lineStart + 1,
        buffer.slice(offset, endOffset).rtrim("\r\n").str(),
    });
    return false;
  }

  const llvm::DenseMap<FileID, std::vector<LineComment>>& comments() const { return comments_; }

 private:
  llvm::DenseMap<FileID, std::vector<LineComment>> comments_;
};

class Indexer {
 public:
  Indexer(ASTContext& context, llvm::raw_ostream& out, Preprocessor& preprocessor,
          const LineCommentCollector& comments, Profile& profile)
      : profile_(profile),
        context_(context),
        sources_(context.getSourceManager()),
        policy_(context.getPrintingPolicy()),
        names_(context),
        out_(out),
        preprocessor_(preprocessor),
        comments_(comments) {}

  void run() {
    walkContext(context_.getTranslationUnitDecl(), "");
    ScopedTimer timer(profile_.markerMs);
    emitMarkerBlocks();
  }

  Profile& profile_;

 private:
  // One normalized absolute path and repository membership per FileID. System
  // headers contribute thousands of decls; without this cache each one would
  // re-run make_absolute / remove_dots only to be rejected by the prefix check.
  const CachedFile& fileInfo(SourceLocation expansionBegin) const {
    FileID id = sources_.getFileID(expansionBegin);
    if (!id.isValid()) {
      static const CachedFile kInvalid;
      return kInvalid;
    }
    auto existing = files_.find(id);
    if (existing != files_.end()) return existing->second;

    CachedFile cached;
    PresumedLoc presumed = sources_.getPresumedLoc(expansionBegin);
    if (presumed.isValid()) {
      llvm::SmallString<256> path(presumed.getFilename());
      llvm::sys::fs::make_absolute(path);
      llvm::sys::path::remove_dots(path, true);
      cached.absolute = path.str().str();
      cached.indexed = inRepository(cached.absolute);
    }
    return files_.try_emplace(id, std::move(cached)).first->second;
  }

  // A declaration's own file decides whether it is indexed at all, so the cost
  // of a toolchain header is one FileID lookup rather than a serialised node.
  Location locate(const Decl* declaration) const {
    SourceRange range = declaration->getSourceRange();
    Location location = locate(sources_.getExpansionLoc(range.getBegin()));
    PresumedLoc end = sources_.getPresumedLoc(sources_.getExpansionLoc(range.getEnd()));
    location.endLine = end.isValid() ? end.getLine() : location.line;
    return location;
  }

  Location locate(SourceLocation sourceLocation) const {
    Location location;
    SourceLocation beginLoc = sources_.getExpansionLoc(sourceLocation);
    const CachedFile& file = fileInfo(beginLoc);
    location.file = file.absolute;
    PresumedLoc begin = sources_.getPresumedLoc(beginLoc);
    if (begin.isValid()) {
      location.line = begin.getLine();
      location.column = begin.getColumn();
    }
    if (beginLoc.isValid() && beginLoc.isFileID()) {
      location.offset = static_cast<int64_t>(sources_.getFileOffset(beginLoc));
    }
    location.endLine = location.line;
    return location;
  }

  std::string declarationUsr(const Decl* declaration) const {
    llvm::SmallString<128> buffer;
    if (index::generateUSRForDecl(declaration->getCanonicalDecl(), buffer)) return "";
    return buffer.str().str();
  }

  // Clang's JSON dump records a type's spelling and, when the top level of that
  // spelling is sugar, its single-step desugaring; reccmp prefers the latter.
  // Nested sugar is deliberately left alone by both: `LPCSTR` becomes
  // `const CHAR *`, not `const char *`.
  std::string typeName(QualType type) const {
    if (type.isNull()) return "";
    SplitQualType spelled = type.split();
    SplitQualType desugared = type.getSplitDesugaredType();
    return QualType::getAsString(desugared != spelled ? desugared : spelled, policy_);
  }

  // The canonical spelling of a type, for identities compared across
  // translation units. Single-step desugaring preserves typedef and elaborated
  // spellings (`W8NavigatorAttachment *` vs `struct W8NavigatorAttachment *`),
  // which describe one type and must compare equal; the canonical spelling
  // dissolves both. The printing policy suppresses the tag keyword in C++
  // but keeps it in C, so the keyword is forced back on: a Windows HANDLE
  // parameter must spell identically whether the including TU is C or C++.
  // Display strings such as source signatures keep the spelled form.
  std::string canonicalName(QualType type) const {
    if (type.isNull()) return "";
    PrintingPolicy canonical(policy_);
    canonical.SuppressTagKeyword = false;
    return QualType::getAsString(type.getCanonicalType().split(), canonical);
  }

  // Pointer layers peeled from the outside of the desugared type, so the
  // depth comes from the type structure rather than counting `*` in a
  // spelling. A reference, array or function type on the outside stops the
  // peel at zero: those spellings never end in `*` either, and their layout
  // comparison would be against a different kind of Ghidra type.
  static int pointerDepth(QualType type) {
    int depth = 0;
    QualType current = type;
    while (!current.isNull()) {
      const auto* pointer = dyn_cast<PointerType>(current.getSplitDesugaredType().Ty);
      if (!pointer) break;
      ++depth;
      current = pointer->getPointeeType();
    }
    return depth;
  }

  void describeStorage(llvm::json::Object& entry, QualType type) const {
    QualType current = type.getCanonicalType();
    if (current->getAs<ReferenceType>()) {
      entry["storage_kind"] = "reference";
      return;
    }
    if (current->getAs<PointerType>() || pointerDepth(type) > 0) {
      entry["storage_kind"] = "pointer";
      return;
    }
      if (const ArrayType* array = current->getAsArrayTypeUnsafe()) {
      QualType element = array->getElementType();
      entry["storage_kind"] = "array";
      entry["array_element_type"] = typeName(element);
      if (!element->isDependentType() && element->isConstantSizeType()) {
        entry["array_stride"] = context_.getTypeSizeInChars(element).getQuantity();
      }
      if (const auto* constant = dyn_cast<ConstantArrayType>(array)) {
        entry["array_count"] = static_cast<int64_t>(constant->getSize().getZExtValue());
      }
      QualType element_canonical = element.getCanonicalType();
      if (element_canonical->getAs<ReferenceType>()) {
        entry["array_element_kind"] = "reference";
      } else if (element_canonical->getAs<PointerType>() || pointerDepth(element) > 0) {
        entry["array_element_kind"] = "pointer";
      } else if (element_canonical->getAsArrayTypeUnsafe()) {
        entry["array_element_kind"] = "array";
      } else if (element_canonical->getAsCXXRecordDecl()) {
        entry["array_element_kind"] = "embedded_record";
      } else {
        entry["array_element_kind"] = "scalar";
      }
      return;
    }
    if (current->getAsCXXRecordDecl()) {
      entry["storage_kind"] = "embedded_record";
      return;
    }
    entry["storage_kind"] = "scalar";
  }

  // Semantic id of the CXX record a type ultimately refers to (after peeling
  // pointers, references and arrays). Empty when the type is not a record.
  // Layout lookup uses this instead of stripping qualifiers from spellings.
  std::string recordSemanticId(QualType type) const {
    QualType current = type.getCanonicalType();
    while (!current.isNull()) {
      if (const auto* reference = current->getAs<ReferenceType>()) {
        current = reference->getPointeeType().getCanonicalType();
        continue;
      }
      if (const auto* pointer = current->getAs<PointerType>()) {
        current = pointer->getPointeeType().getCanonicalType();
        continue;
      }
      if (const ArrayType* array = current->getAsArrayTypeUnsafe()) {
        current = array->getElementType().getCanonicalType();
        continue;
      }
      break;
    }
    const CXXRecordDecl* record = current->getAsCXXRecordDecl();
    if (!record || !record->getIdentifier()) return "";
    std::string qualified;
    llvm::raw_string_ostream stream(qualified);
    record->printQualifiedName(stream, policy_);
    return "record:" + stream.str();
  }

  std::string templateArguments(const ClassTemplateSpecializationDecl* specialization) const {
    std::vector<std::string> rendered;
    for (const TemplateArgument& argument : specialization->getTemplateArgs().asArray()) {
      std::string text;
      switch (argument.getKind()) {
        case TemplateArgument::Type:
          text = typeName(argument.getAsType());
          break;
        case TemplateArgument::Integral:
          text = llvm::toString(argument.getAsIntegral(), 10, true);
          break;
        default:
          break;
      }
      if (!text.empty()) rendered.push_back(text);
    }
    return join(rendered, ", ");
  }

  // The name a scope contributes to a qualified name. A specialization carries
  // its arguments; an unnamed namespace or record contributes nothing, which is
  // why an anonymous-namespace function is indexed under its bare name.
  std::string component(const Decl* declaration) const {
    if (const auto* specialization = dyn_cast<ClassTemplateSpecializationDecl>(declaration)) {
      std::string arguments = templateArguments(specialization);
      std::string name = specialization->getNameAsString();
      return arguments.empty() ? name : name + "<" + arguments + ">";
    }
    if (const auto* record = dyn_cast<CXXRecordDecl>(declaration)) {
      return record->getIdentifier() ? record->getNameAsString() : "";
    }
    if (const auto* space = dyn_cast<NamespaceDecl>(declaration)) {
      return space->getIdentifier() ? space->getNameAsString() : "";
    }
    return "";
  }

  // The semantic scope, which is what an out-of-line member definition must be
  // indexed under. Contexts that are not records or named namespaces - function
  // bodies, linkage specifications, unnamed namespaces - contribute nothing.
  std::string scopeOf(const DeclContext* context) const {
    std::vector<std::string> parts;
    for (const DeclContext* node = context; node && !node->isTranslationUnit();
         node = node->getParent()) {
      if (!isa<CXXRecordDecl>(node) && !isa<NamespaceDecl>(node)) continue;
      std::string part = component(cast<Decl>(node));
      if (!part.empty()) parts.push_back(part);
    }
    std::vector<std::string> ordered(parts.rbegin(), parts.rend());
    return join(ordered, "::");
  }

  // A dependent declaration has no mangled name, so the index falls back to the
  // declaration kind, qualified name, and canonical function type — the identity
  // reccmp uses for an uninstantiated template pattern. The canonical type
  // carries cv-qualifiers, ref-qualifiers, and variadic-ness in one signature,
  // so overloads that differ only there do not collide.
  std::string semanticId(const FunctionDecl* function, llvm::StringRef qualifiedName) const {
    bool manglable =
        !function->isDependentContext() && !function->getDescribedFunctionTemplate();
    if (manglable) {
      std::string mangled = names_.getName(function);
      if (!mangled.empty()) return mangled;
    }
    // Qualified, because a FunctionDecl is both a Decl and a DeclContext.
    return (llvm::Twine(function->Decl::getDeclKindName()) + "Decl:" + qualifiedName + ":" +
            canonicalName(function->getType()))
        .str();
  }

  // A function's semantic id, as its declaration record states it.
  std::string functionIdentity(const FunctionDecl* function) const {
    return semanticId(
        function, qualify(scopeOf(function->getDeclContext()), function->getNameAsString()));
  }

  std::vector<std::string> parameterTypes(const FunctionDecl* function) const {
    std::vector<std::string> parameters;
    for (const ParmVarDecl* parameter : function->parameters()) {
      parameters.push_back(canonicalName(parameter->getType()));
    }
    return parameters;
  }

  static bool hasThis(llvm::StringRef semanticKind) {
    return semanticKind == "constructor" || semanticKind == "destructor" ||
           semanticKind == "instance_method";
  }

  // The convention Clang assigned: spelled, or the target default for the
  // kind of function (a variadic member function is __cdecl).
  static std::string callingConvention(const FunctionDecl* function) {
    switch (function->getType()->castAs<FunctionType>()->getCallConv()) {
      case CC_X86StdCall: return "__stdcall";
      case CC_X86FastCall: return "__fastcall";
      case CC_X86ThisCall: return "__thiscall";
      case CC_X86VectorCall: return "__vectorcall";
      default: return "__cdecl";
    }
  }

  // Linkage as computed by Clang, spelled for the index. Consumers that join
  // declarations across translation units need the raw internal linkage, not
  // the formal one: entities in an anonymous namespace are formally external
  // but TU-local, and unrelated TU-local `static` definitions must never be
  // joined by their shared spelling.
  static std::string linkageName(Linkage linkage) {
    switch (linkage) {
      case Linkage::Invalid:
        return "invalid";
      case Linkage::None:
        return "none";
      case Linkage::Internal:
        return "internal";
      case Linkage::UniqueExternal:
        return "unique-external";
      case Linkage::VisibleNone:
        return "visible-none";
      case Linkage::Module:
        return "module";
      case Linkage::External:
        return "external";
    }
    return "invalid";
  }

  // The storage class as written in the source. Whether a declaration is
  // TU-local is decided by the computed linkage above, not by this spelling:
  // a `constexpr` global has no storage class but still has internal linkage.
  static std::string storageClassName(StorageClass storage) {
    switch (storage) {
      case SC_None:
        return "none";
      case SC_Extern:
        return "extern";
      case SC_Static:
        return "static";
      case SC_PrivateExtern:
        return "private-extern";
      case SC_Auto:
        return "auto";
      case SC_Register:
        return "register";
    }
    return "invalid";
  }

  // A namespace-scope `int x;` without an initializer is only a tentative
  // definition under C linkage rules; it still defines common storage, so the
  // index ranks it above a pure declaration but below an initialized one.
  static std::string definitionKindName(VarDecl::DefinitionKind kind) {
    switch (kind) {
      case VarDecl::DeclarationOnly:
        return "declaration";
      case VarDecl::TentativeDefinition:
        return "tentative";
      case VarDecl::Definition:
        return "definition";
    }
    return "invalid";
  }

  std::string sourceSignature(const FunctionDecl* function, llvm::StringRef qualifiedName,
                              llvm::StringRef semanticKind, llvm::StringRef returnType,
                              llvm::StringRef convention) const {
    std::string signature;
    if (semanticKind != "constructor" && semanticKind != "destructor") {
      signature += returnType.str();
      signature += " ";
      if (convention != "__cdecl" && convention != "__thiscall") {
        signature += convention.str();
        signature += " ";
      }
    }
    signature += qualifiedName.str();
    signature += "(";
    for (unsigned index = 0; index < function->getNumParams(); ++index) {
      if (index) signature += ", ";
      const ParmVarDecl* parameter = function->getParamDecl(index);
      signature += typeName(parameter->getOriginalType());
      if (!parameter->getName().empty()) {
        signature += " ";
        signature += parameter->getNameAsString();
      }
    }
    if (function->isVariadic()) {
      if (function->getNumParams()) signature += ", ";
      signature += "...";
    }
    signature += ")";
    if (const auto* method = dyn_cast<CXXMethodDecl>(function); method && method->isConst()) {
      signature += " const";
    }
    return signature;
  }

  // The identity a field has in every unit: its owner's USR, declaration
  // position and index.
  std::string fieldIdentity(const FieldDecl* field) const {
    const auto* owner = dyn_cast<CXXRecordDecl>(field->getParent());
    if (owner) {
      const auto* definition = dyn_cast_or_null<CXXRecordDecl>(owner->getDefinition());
      owner = definition ? definition : owner->getCanonicalDecl();
    }
    std::string ownerIdentity = owner ? declarationUsr(owner) : "";
    if (ownerIdentity.empty()) ownerIdentity = "unknown-owner";
    Location location = locate(field);
    return ownerIdentity + "::field@" + relative(location.file) + ":" +
           std::to_string(location.line) + ":" + std::to_string(location.column) + ":" +
           std::to_string(field->getFieldIndex());
  }

  // Width in bits and signedness of an integer, enumeration or pointer value.
  std::optional<std::pair<int64_t, bool>> integerShape(QualType type) const {
    if (type.isNull() || type->isDependentType() || type->isIncompleteType()) return {};
    QualType canonical = type.getCanonicalType();
    if (canonical->isIntegralOrEnumerationType()) {
      return std::make_pair(static_cast<int64_t>(context_.getTypeSize(canonical)),
                            canonical->isSignedIntegerOrEnumerationType());
    }
    if (canonical->isPointerType()) {
      return std::make_pair(static_cast<int64_t>(context_.getTypeSize(canonical)), false);
    }
    return {};
  }

  // Register footprint of a returned value: void, i8/i16/i32/i64, float, or
  // unknown (aggregates, whose return convention this does not decide).
  std::string returnKind(QualType type) const {
    if (type.isNull() || type->isDependentType()) return "unknown";
    QualType canonical = type.getCanonicalType();
    if (canonical->isVoidType()) return "void";
    if (canonical->isRealFloatingType()) return "float";
    if (auto shape = integerShape(canonical)) {
      switch (shape->first) {
        case 8: return "i8";
        case 16: return "i16";
        case 32: return "i32";
        case 64: return "i64";
        default: return "unknown";
      }
    }
    if (canonical->isReferenceType()) return "i32";
    return "unknown";
  }

  // What a caller may assume about calling `function` on 32-bit x86 under
  // the Microsoft ABI: register arguments, the argument bytes the callee
  // removes, and the return kind. Unknown fields are null.
  llvm::json::Value callFacts(const FunctionDecl* function, llvm::StringRef convention,
                              llvm::StringRef semanticKind) const {
    if (function->isDependentContext() || function->getDescribedFunctionTemplate()) {
      return nullptr;
    }
    bool variadic = function->isVariadic();
    // A variadic member function is __cdecl with `this` on the stack.
    bool thiscall = convention == "__thiscall";
    bool fastcall = convention == "__fastcall";
    bool calleePops = !variadic && (convention == "__stdcall" || thiscall || fastcall);
    bool usesEcx = thiscall;
    bool usesEdx = false;
    int64_t stack = 0;
    bool stackKnown = true;
    int fastcallRegisters = 0;
    if (hasThis(semanticKind)) {
      if (fastcall) {
        usesEcx = true;  // `this` is the first register argument
        fastcallRegisters = 1;
      } else if (!thiscall) {
        stack += 4;
      }
    }
    QualType returned = function->getReturnType().getCanonicalType();
    if (!returned->isVoidType() && returned->getAsCXXRecordDecl()) {
      // Whether a record comes back in registers or through a hidden
      // pointer argument is not decided here.
      stackKnown = false;
    }
    for (const ParmVarDecl* parameter : function->parameters()) {
      QualType type = parameter->getType().getCanonicalType();
      if (type->isDependentType() || type->isIncompleteType()) {
        stackKnown = false;
        continue;
      }
      int64_t size = type->isReferenceType()
                         ? 4
                         : context_.getTypeSizeInChars(type).getQuantity();
      bool inRegister = fastcall && fastcallRegisters < 2 && size <= 4 &&
                        (type->isIntegralOrEnumerationType() || type->isPointerType() ||
                         type->isReferenceType());
      if (inRegister) {
        (fastcallRegisters++ == 0 ? usesEcx : usesEdx) = true;
        continue;
      }
      if (type->getAsCXXRecordDecl() && !type->getAsCXXRecordDecl()->isTrivial()) {
        stackKnown = false;  // passed by address or with a temporary copy
      }
      stack += (size + 3) / 4 * 4;
    }
    llvm::json::Value cleanup = nullptr;
    if (!calleePops) cleanup = 0;
    else if (stackKnown) cleanup = stack;
    return llvm::json::Object{
        {"uses_ecx", usesEcx},
        {"uses_edx", usesEdx},
        {"stack_cleanup", std::move(cleanup)},
        {"return_kind", semanticKind == "constructor" || semanticKind == "destructor"
                            ? std::string("unknown")
                            : returnKind(function->getReturnType())},
    };
  }

  void emitDeclaration(const FunctionDecl* function, const Location& location) {
    const DeclContext* context = function->getDeclContext();
    bool isMember = isa<CXXRecordDecl>(context);
    std::string scope = scopeOf(context);
    std::string qualifiedName = qualify(scope, function->getNameAsString());
    std::string functionIdentity = semanticId(function, qualifiedName);

    std::string semanticKind;
    if (isa<CXXConstructorDecl>(function)) {
      semanticKind = "constructor";
    } else if (isa<CXXDestructorDecl>(function)) {
      semanticKind = "destructor";
    } else if (isMember) {
      // isStatic covers implicitly static members (operator new/delete).
      const auto* method = dyn_cast<CXXMethodDecl>(function);
      semanticKind = method && method->isStatic() ? "static_method" : "instance_method";
    } else {
      semanticKind = scope.empty() ? "free_function" : "namespace_function";
    }

    std::string functionType = typeName(function->getType());
    // The record's compared identity dissolves typedef and elaborated
    // spellings; the display signature keeps the spelled form.
    std::string returnType;
    std::string spelledReturn;
    if (semanticKind != "constructor" && semanticKind != "destructor") {
      returnType = canonicalName(function->getReturnType());
      spelledReturn =
          llvm::StringRef(functionType).take_until([](char c) { return c == '('; }).trim().str();
    }

    std::vector<std::string> parameters = parameterTypes(function);
    llvm::json::Array parameterTypes;
    for (const std::string& parameter : parameters) parameterTypes.push_back(parameter);
    llvm::json::Array parameterReferences;
    llvm::json::Array parameterReferenceForms;
    for (const ParmVarDecl* parameter : function->parameters()) {
      QualType original = parameter->getOriginalType();
      parameterReferences.push_back(original->isReferenceType());
      std::string kind = "value";
      QualType referred;
      if (const auto* reference = original->getAs<LValueReferenceType>()) {
        referred = reference->getPointeeType();
        if (referred->isPointerType()) kind = "lvalue-reference-to-pointer";
        else if (referred->isArrayType()) kind = "lvalue-reference-to-array";
        else if (referred->isFunctionType()) kind = "lvalue-reference-to-function";
        else kind = "lvalue-reference-to-object";
      } else if (const auto* reference = original->getAs<RValueReferenceType>()) {
        referred = reference->getPointeeType();
        if (referred->isPointerType()) kind = "rvalue-reference-to-pointer";
        else if (referred->isArrayType()) kind = "rvalue-reference-to-array";
        else if (referred->isFunctionType()) kind = "rvalue-reference-to-function";
        else kind = "rvalue-reference-to-object";
      }
      parameterReferenceForms.push_back(llvm::json::Object{
          {"kind", kind},
          {"const", !referred.isNull() && referred.isConstQualified()},
      });
    }

    std::string convention = callingConvention(function);
    llvm::json::Object record{
        {"record", "declaration"},
        {"semantic_id", functionIdentity},
        {"qualified_name", qualifiedName},
        {"semantic_kind", semanticKind},
        {"calling_convention", convention},
        {"linkage", linkageName(function->getLinkageInternal())},
        {"storage_class", storageClassName(function->getStorageClass())},
        {"source_signature",
         sourceSignature(function, qualifiedName, semanticKind, spelledReturn, convention)},
        {"parameter_references", std::move(parameterReferences)},
        {"parameter_reference_forms", std::move(parameterReferenceForms)},
        {"return_type", returnType},
        {"parameter_types", std::move(parameterTypes)},
        {"owning_class", isMember ? llvm::json::Value(scope) : llvm::json::Value(nullptr)},
        {"has_this", hasThis(semanticKind)},
        {"is_virtual", isVirtual(function)},
        {"is_variadic", function->isVariadic()},
        {"source_file", relative(location.file)},
        {"line", location.line},
        {"end_line", location.endLine},
        // A primary template body is source for its dependent member uses, but
        // its definition is not a concrete emitted function that owns a reccmp
        // FUNCTION marker. Keep its marker-join row declaration-only; an emitted
        // specialization carries the concrete function identity and extent.
        {"is_definition", isEmittedDefinition(function)},
        {"call", callFacts(function, convention, semanticKind)},
    };
    emit(std::move(record));
    emitMemberUses(function, functionIdentity, qualifiedName, location);
  }

  // One variable definition or declaration. Parameters are VarDecls too,
  // but they never reach this emitter: the walker filters them out, along
  // with implicit declarations such as a function body's `__func__`.
  // Mangling a variable in a dependent context is meaningless, so those fall
  // back to a qualified signature identity, mirroring uninstantiated
  // template patterns for functions.
  std::string variableSemanticId(const VarDecl* variable, llvm::StringRef qualifiedName) const {
    std::string mangled;
    if (!variable->getDeclContext()->isDependentContext()) mangled = names_.getName(variable);
    if (!mangled.empty()) return mangled;
    return ("VarDecl:" + qualifiedName + "(" + canonicalName(variable->getType()) + ")").str();
  }

  void emitVariable(const VarDecl* variable, const Location& location) {
    const DeclContext* context = variable->getDeclContext();
    std::string scope = scopeOf(context);
    std::string qualifiedName = qualify(scope, variable->getNameAsString());
    std::string type = canonicalName(variable->getType());
    llvm::json::Object payload{
        {"record", "variable"},
        {"semantic_id", variableSemanticId(variable, qualifiedName)},
        {"qualified_name", qualifiedName},
        {"type", type},
        {"linkage", linkageName(variable->getLinkageInternal())},
        {"storage_class", storageClassName(variable->getStorageClass())},
        {"definition_kind",
         definitionKindName(variable->isThisDeclarationADefinition())},
        {"source_file", relative(location.file)},
        {"line", location.line},
        {"end_line", location.endLine},
    };
    std::string recordId = recordSemanticId(variable->getType());
    if (!recordId.empty()) payload["record_semantic_id"] = recordId;
    emit(std::move(payload));
  }

  static bool isVirtual(const FunctionDecl* function) {
    const auto* method = dyn_cast<CXXMethodDecl>(function);
    return method && method->isVirtual();
  }

  static bool isEmittedDefinition(const FunctionDecl* function) {
    return function->doesThisDeclarationHaveABody() && !function->isLateTemplateParsed() &&
           !function->getDescribedFunctionTemplate();
  }

  class MemberUseVisitor : public RecursiveASTVisitor<MemberUseVisitor> {
   public:
    MemberUseVisitor(Indexer& indexer, llvm::StringRef functionIdentity,
                     llvm::StringRef functionName,
                     const Location& functionLocation)
        : indexer_(indexer),
          functionIdentity_(functionIdentity),
          functionName_(functionName),
          functionLocation_(functionLocation) {}

    bool VisitMemberExpr(MemberExpr* expression) {
      const auto* field = dyn_cast<FieldDecl>(expression->getMemberDecl());
      if (field) emitResolved(expression, field);
      return true;
    }

    bool VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr* expression) {
      Location useLocation = indexer_.locate(expression->getMemberLoc());
      std::string owner = indexer_.typeName(expression->getBaseType());
      std::string name = expression->getMember().getAsString();
      emit(nullptr, expression, useLocation, owner, name, "", "", nullptr);
      return true;
    }

    // Every call the body makes: the callee's identity (or, for a virtual
    // call, the declaration that introduces the vtable slot), and which
    // arguments are plain field reads.
    bool VisitCallExpr(CallExpr* call) {
      llvm::json::Object entry;
      const FunctionDecl* callee = call->getDirectCallee();
      entry["callee"] = callee ? llvm::json::Value(indexer_.functionIdentity(callee))
                               : llvm::json::Value(nullptr);
      const auto* member = dyn_cast<CXXMemberCallExpr>(call);
      const CXXMethodDecl* method = member ? member->getMethodDecl() : nullptr;
      const auto* access =
          member ? dyn_cast<MemberExpr>(member->getCallee()->IgnoreParens()) : nullptr;
      bool isVirtual = method && method->isVirtual() && !(access && access->hasQualifier());
      entry["virtual"] = isVirtual;
      if (isVirtual) {
        const CXXMethodDecl* slot = method;
        while (slot->size_overridden_methods() > 0) slot = *slot->begin_overridden_methods();
        entry["slot"] = indexer_.functionIdentity(slot);
        entry["object_class"] = indexer_.recordSemanticId(
            member->getImplicitObjectArgument()->getType());
      }
      if (member) {
        entry["object"] = baseKind(member->getImplicitObjectArgument());
      }
      llvm::json::Array arguments;
      for (const Expr* argument : call->arguments()) {
        const auto* read = dyn_cast<MemberExpr>(argument->IgnoreParenImpCasts());
        const auto* field = read ? dyn_cast<FieldDecl>(read->getMemberDecl()) : nullptr;
        arguments.push_back(field ? llvm::json::Value(indexer_.fieldIdentity(field))
                                  : llvm::json::Value(nullptr));
      }
      entry["field_arguments"] = std::move(arguments);
      Location where = indexer_.locate(call->getBeginLoc());
      entry["line"] = where.line;
      entry["offset"] = where.offset >= 0 ? llvm::json::Value(where.offset)
                                          : llvm::json::Value(nullptr);
      calls_.push_back(std::move(entry));
      return true;
    }

    llvm::json::Array takeCalls() { return std::move(calls_); }

   private:
    // What the object of a member access or call is: this, a parameter
    // (with its index), a local, a global, another member, or other.
    llvm::json::Object baseKind(const Expr* base) const {
      if (!base) return llvm::json::Object{{"kind", "this"}};
      base = base->IgnoreParenImpCasts();
      if (isa<CXXThisExpr>(base)) return llvm::json::Object{{"kind", "this"}};
      if (const auto* reference = dyn_cast<DeclRefExpr>(base)) {
        if (const auto* parameter = dyn_cast<ParmVarDecl>(reference->getDecl())) {
          return llvm::json::Object{
              {"kind", "parameter"},
              {"index", static_cast<int64_t>(parameter->getFunctionScopeIndex())}};
        }
        if (const auto* variable = dyn_cast<VarDecl>(reference->getDecl())) {
          return llvm::json::Object{{"kind", variable->hasLocalStorage() ? "local" : "global"}};
        }
      }
      if (isa<MemberExpr>(base)) return llvm::json::Object{{"kind", "member"}};
      return llvm::json::Object{{"kind", "other"}};
    }

    static bool contains(const Stmt* root, const Stmt* sought) {
      if (!root) return false;
      if (root == sought) return true;
      for (const Stmt* child : root->children()) {
        if (contains(child, sought)) return true;
      }
      return false;
    }

    static bool transparent(const Stmt* statement) {
      return isa<CastExpr>(statement) || isa<ParenExpr>(statement) ||
             isa<ExprWithCleanups>(statement) ||
             isa<MaterializeTemporaryExpr>(statement) ||
             isa<CXXBindTemporaryExpr>(statement) || isa<ConstantExpr>(statement) ||
             isa<CXXDefaultArgExpr>(statement) || isa<CXXDefaultInitExpr>(statement);
    }

    std::vector<const Stmt*> parents(const Stmt* statement) const {
      std::vector<const Stmt*> result;
      DynTypedNode current = DynTypedNode::create(*statement);
      for (unsigned depth = 0; depth < 128; ++depth) {
        DynTypedNodeList found = indexer_.context_.getParents(current);
        if (found.empty()) break;
        const Stmt* parent = nullptr;
        for (const DynTypedNode& candidate : found) {
          if ((parent = candidate.get<Stmt>())) break;
        }
        if (!parent) break;
        result.push_back(parent);
        current = DynTypedNode::create(*parent);
      }
      return result;
    }

    static bool isNonEvaluated(const std::vector<const Stmt*>& ancestors) {
      for (const Stmt* ancestor : ancestors) {
        if (const auto* unary = dyn_cast<UnaryExprOrTypeTraitExpr>(ancestor)) {
          if (unary->getKind() == UETT_SizeOf || unary->getKind() == UETT_AlignOf)
            return true;
        }
        if (isa<CXXNoexceptExpr>(ancestor) || isa<TypeTraitExpr>(ancestor)) return true;
      }
      return false;
    }

    static std::string integerValue(const Expr* expression) {
      expression = expression->IgnoreParenImpCasts();
      bool negative = false;
      if (const auto* unary = dyn_cast<UnaryOperator>(expression)) {
        if (unary->getOpcode() == UO_Minus) {
          negative = true;
          expression = unary->getSubExpr()->IgnoreParenImpCasts();
        }
      }
      const auto* integer = dyn_cast<IntegerLiteral>(expression);
      if (!integer) return "";
      std::string value = llvm::toString(integer->getValue(), 10, true);
      return negative ? "-" + value : value;
    }

    void addArrayIndex(const ArraySubscriptExpr* subscript,
                       llvm::json::Array& indices) const {
      std::string value = integerValue(subscript->getIdx());
      llvm::json::Object index{{"constant", !value.empty()}};
      if (!value.empty()) index["value"] = value;
      indices.push_back(std::move(index));
    }

    static bool isAssignmentOperatorOverload(const CXXOperatorCallExpr* call) {
      OverloadedOperatorKind kind = call->getOperator();
      return kind == OO_Equal || kind == OO_PlusEqual || kind == OO_MinusEqual ||
             kind == OO_StarEqual || kind == OO_SlashEqual || kind == OO_PercentEqual ||
             kind == OO_AmpEqual || kind == OO_PipeEqual || kind == OO_CaretEqual ||
             kind == OO_LessLessEqual || kind == OO_GreaterGreaterEqual;
    }

    static bool isCompoundAssignment(OverloadedOperatorKind kind) {
      return kind != OO_Equal;
    }

    void classifyMemoryCall(const CallExpr* call, const Expr* member,
                            std::set<std::string>& operations) const {
      const FunctionDecl* callee = call->getDirectCallee();
      if (!callee) return;
      std::string name = callee->getQualifiedNameAsString();
      std::string lowered = llvm::StringRef(name).lower();
      int destination = -1;
      int source = -1;
      if (lowered == "memcpy" || lowered == "std::memcpy" || lowered == "memmove" ||
          lowered == "std::memmove" || lowered == "strcpy" || lowered == "strncpy") {
        destination = 0;
        source = 1;
      } else if (lowered == "memcpy_s" || lowered == "memmove_s") {
        destination = 0;
        source = 2;
      } else if (lowered == "memset" || lowered == "std::memset") {
        destination = 0;
      } else if (lowered == "std::copy" || lowered == "copy" ||
                 lowered == "std::copy_n" || lowered == "copy_n") {
        source = 0;
        destination = 2;
      } else if (lowered == "std::copy_backward" || lowered == "copy_backward") {
        source = 0;
        destination = 2;
      }
      if (destination >= 0 && static_cast<unsigned>(destination) < call->getNumArgs() &&
          contains(call->getArg(destination), member)) {
        operations.insert("copy-memory-destination");
      }
      if (source >= 0 && static_cast<unsigned>(source) < call->getNumArgs() &&
          contains(call->getArg(source), member)) {
        operations.insert("copy-memory-source");
      }
    }

    void emitResolved(const MemberExpr* expression, const FieldDecl* field) {
      const CXXRecordDecl* owner = dyn_cast<CXXRecordDecl>(field->getParent());
      if (owner) {
        const CXXRecordDecl* definition =
            dyn_cast<CXXRecordDecl>(owner->getDefinition());
        owner = definition ? definition : owner->getCanonicalDecl();
      }
      std::string ownerName;
      if (owner) {
        llvm::raw_string_ostream stream(ownerName);
        owner->printQualifiedName(stream, indexer_.policy_);
        stream.flush();
      }
      Location declarationLocation = indexer_.locate(field);
      Location useLocation = indexer_.locate(expression->getMemberLoc());
      const std::string ownerIdentity = owner ? indexer_.declarationUsr(owner) : "";
      const std::string fieldUsr = indexer_.declarationUsr(field);
      emit(field, expression, useLocation, ownerName, field->getNameAsString(),
           ownerIdentity, fieldUsr, &declarationLocation, indexer_.fieldIdentity(field));
    }

    void emit(const FieldDecl* field, const Expr* expression,
              const Location& useLocation, llvm::StringRef ownerName,
              llvm::StringRef fieldName, llvm::StringRef ownerIdentity,
              llvm::StringRef fieldUsr, const Location* declarationLocation,
              llvm::StringRef fieldIdentity = "") {
      std::vector<const Stmt*> ancestors = parents(expression);
      std::set<std::string> operations;
      llvm::json::Array arrayIndices;
      llvm::json::Array conversions;
      std::string conversionSource = indexer_.canonicalName(expression->getType());
      QualType conversionSourceType = expression->getType();
      for (const Stmt* ancestor : ancestors) {
        if (const auto* cast = dyn_cast<CastExpr>(ancestor)) {
          std::string destination = indexer_.canonicalName(cast->getType());
          if (destination != conversionSource) {
            llvm::json::Object conversion{
                {"kind", cast->getCastKindName()},
                {"source_type", conversionSource},
                {"destination_type", destination},
            };
            auto from = indexer_.integerShape(conversionSourceType);
            auto to = indexer_.integerShape(cast->getType());
            if (from && to) {
              conversion["source_bits"] = from->first;
              conversion["source_signed"] = from->second;
              conversion["destination_bits"] = to->first;
            }
            conversions.push_back(std::move(conversion));
          }
          conversionSource = destination;
          conversionSourceType = cast->getType();
        }
        if (const auto* subscript = dyn_cast<ArraySubscriptExpr>(ancestor)) {
          if (contains(subscript->getBase(), expression)) {
            operations.insert("array-index");
            addArrayIndex(subscript, arrayIndices);
          }
        }
        if (const auto* unary = dyn_cast<UnaryOperator>(ancestor)) {
          if (!contains(unary->getSubExpr(), expression)) continue;
          if (unary->getOpcode() == UO_AddrOf) operations.insert("address-taken");
          if (unary->isIncrementDecrementOp()) {
            operations.insert("read");
            operations.insert("write");
          }
        }
        if (const auto* binary = dyn_cast<BinaryOperator>(ancestor)) {
          if (!binary->isAssignmentOp()) continue;
          if (contains(binary->getLHS(), expression)) {
            operations.insert("write");
            if (binary->isCompoundAssignmentOp()) operations.insert("read");
            if (binary->getLHS()->IgnoreParenImpCasts() == expression &&
                (expression->getType()->isRecordType() || expression->getType()->isArrayType())) {
              operations.insert("copy-memory-destination");
            }
          }
          if (contains(binary->getRHS(), expression)) {
            operations.insert("read");
            if (binary->getRHS()->IgnoreParenImpCasts() == expression &&
                (expression->getType()->isRecordType() || expression->getType()->isArrayType())) {
              operations.insert("copy-memory-source");
            }
          }
        }
        if (const auto* overload = dyn_cast<CXXOperatorCallExpr>(ancestor)) {
          if (isAssignmentOperatorOverload(overload) && overload->getNumArgs() > 1) {
            if (contains(overload->getArg(0), expression)) {
              operations.insert("write");
              if (isCompoundAssignment(overload->getOperator())) operations.insert("read");
              if (overload->getOperator() == OO_Equal &&
                  overload->getArg(0)->IgnoreParenImpCasts() == expression &&
                  expression->getType()->isRecordType()) {
                operations.insert("copy-memory-destination");
              }
            }
            if (contains(overload->getArg(1), expression)) {
              operations.insert("read");
              if (overload->getOperator() == OO_Equal &&
                  overload->getArg(1)->IgnoreParenImpCasts() == expression &&
                  expression->getType()->isRecordType()) {
                operations.insert("copy-memory-source");
              }
            }
          }
        }
        if (const auto* call = dyn_cast<CallExpr>(ancestor)) {
          classifyMemoryCall(call, expression, operations);
        }
        if (const auto* outerMember = dyn_cast<MemberExpr>(ancestor)) {
          if (contains(outerMember->getBase(), expression)) operations.insert("member-base");
        }
      }

      if (isNonEvaluated(ancestors)) operations.insert("unevaluated");
      const bool hasArrayUse = operations.find("array-index") != operations.end();
      const bool hasWrite = operations.find("write") != operations.end();
      const bool hasAddress = operations.find("address-taken") != operations.end();
      const bool memberBase = operations.find("member-base") != operations.end();
      if (hasArrayUse && !hasWrite) operations.insert("read");
      if (!hasWrite && !hasAddress && !memberBase &&
          operations.find("read") == operations.end() &&
          operations.find("unevaluated") == operations.end()) {
        operations.insert("read");
      }

      llvm::json::Array operationList;
      for (const std::string& operation : operations) operationList.push_back(operation);

      int64_t offsetBits = -1;
      int64_t extentBits = -1;
      if (field) {
        const CXXRecordDecl* owner = dyn_cast<CXXRecordDecl>(field->getParent());
        if (owner) {
          const CXXRecordDecl* definition =
              dyn_cast<CXXRecordDecl>(owner->getDefinition());
          owner = definition ? definition : owner;
        }
        if (owner && !owner->isDependentType() && owner->isCompleteDefinition() &&
            !field->isInvalidDecl()) {
          const ASTRecordLayout& layout = indexer_.context_.getASTRecordLayout(owner);
          offsetBits = static_cast<int64_t>(layout.getFieldOffset(field->getFieldIndex()));
          if (field->isBitField()) {
            extentBits = static_cast<int64_t>(field->getBitWidthValue(indexer_.context_));
          } else if (!field->getType()->isIncompleteType() &&
                     field->getType()->isConstantSizeType()) {
            extentBits = static_cast<int64_t>(indexer_.context_.getTypeSize(field->getType()));
          }
        }
      }
      llvm::json::Object record{
          {"record", "member-use"},
          {"owner_identity", ownerIdentity.empty() ? llvm::json::Value(nullptr)
                                                   : llvm::json::Value(ownerIdentity.str())},
          {"owner_status", ownerIdentity.empty() ? "unknown" : "resolved"},
          {"owner", ownerName.str()},
          {"field_identity", fieldIdentity.empty()
                                 ? ("unknown-owner::field@" + relative(useLocation.file) + ":" +
                                    std::to_string(useLocation.line) + ":" +
                                    std::to_string(useLocation.column) + ":" + fieldName.str())
                                 : fieldIdentity.str()},
          {"field_usr", fieldUsr.empty() ? llvm::json::Value(nullptr)
                                         : llvm::json::Value(fieldUsr.str())},
          {"name", fieldName.str()},
          {"declaration_file", declarationLocation ? relative(declarationLocation->file) : ""},
          {"declaration_line", declarationLocation ? declarationLocation->line : 0},
          {"declaration_column", declarationLocation ? declarationLocation->column : 0},
          {"declaration_offset", declarationLocation ? llvm::json::Value(declarationLocation->offset)
                                                       : llvm::json::Value(nullptr)},
          {"offset_bits", offsetBits >= 0 ? llvm::json::Value(offsetBits)
                                          : llvm::json::Value(nullptr)},
          {"extent_bits", extentBits >= 0 ? llvm::json::Value(extentBits)
                                          : llvm::json::Value(nullptr)},
          {"offset_bytes", offsetBits >= 0 && offsetBits % 8 == 0
                               ? llvm::json::Value(offsetBits / 8)
                               : llvm::json::Value(nullptr)},
          {"extent_bytes", extentBits >= 0 && extentBits % 8 == 0
                               ? llvm::json::Value(extentBits / 8)
                               : llvm::json::Value(nullptr)},
          {"declared_type", field ? indexer_.canonicalName(field->getType())
                                    : indexer_.canonicalName(expression->getType())},
          {"function_identity", functionIdentity_.str()},
          {"function", functionName_.str()},
          {"function_file", relative(functionLocation_.file)},
          {"function_line", functionLocation_.line},
          {"use_file", relative(useLocation.file)},
          {"use_line", useLocation.line},
          {"use_column", useLocation.column},
          {"use_offset", useLocation.offset >= 0 ? llvm::json::Value(useLocation.offset)
                                                 : llvm::json::Value(nullptr)},
          {"operations", std::move(operationList)},
          {"array_indices", std::move(arrayIndices)},
          {"conversions", std::move(conversions)},
          {"base", baseKind(accessBase(expression))},
      };
      indexer_.emit(std::move(record));
    }

    static const Expr* accessBase(const Expr* expression) {
      if (const auto* member = dyn_cast<MemberExpr>(expression)) return member->getBase();
      if (const auto* dependent = dyn_cast<CXXDependentScopeMemberExpr>(expression)) {
        return dependent->isImplicitAccess() ? nullptr : dependent->getBase();
      }
      return nullptr;
    }

    Indexer& indexer_;
    llvm::StringRef functionIdentity_;
    llvm::StringRef functionName_;
    Location functionLocation_;
    llvm::json::Array calls_;
  };

  void emitMemberUses(const FunctionDecl* function, llvm::StringRef functionIdentity,
                      llvm::StringRef functionName, const Location& functionLocation) {
    if (!function->doesThisDeclarationHaveABody()) return;
    ScopedTimer timer(profile_.memberUseMs);
    MemberUseVisitor visitor(*this, functionIdentity, functionName, functionLocation);
    visitor.TraverseStmt(function->getBody());
    llvm::json::Array calls = visitor.takeCalls();
    if (!calls.empty()) {
      emit(llvm::json::Object{
          {"record", "function-facts"},
          {"function", functionIdentity.str()},
          {"calls", std::move(calls)},
      });
    }
  }

  void emitClass(const CXXRecordDecl* record, llvm::StringRef qualifiedName,
                 const Location& location) {
    llvm::json::Array bases;
    for (const CXXBaseSpecifier& base : record->bases()) bases.push_back(typeName(base.getType()));

    const ASTRecordLayout* layout = nullptr;
    if (record->isCompleteDefinition() && !record->isDependentType()) {
      layout = &context_.getASTRecordLayout(record);
    }

    llvm::json::Array fields;
    for (const FieldDecl* field : record->fields()) {
      if (!field->getIdentifier()) continue;
      Location where = locate(field);
      llvm::json::Object entry{
          {"name", field->getNameAsString()},
          {"type", typeName(field->getType())},
          {"pointer_depth", pointerDepth(field->getType())},
          {"source_file", relative(where.file)},
          {"line", where.line},
      };
      if (layout) {
        const uint64_t bitOffset = layout->getFieldOffset(field->getFieldIndex());
        entry["offset"] = static_cast<int64_t>(bitOffset / 8);
        entry["size"] = context_.getTypeSizeInChars(field->getType()).getQuantity();
        if (field->isBitField()) {
          entry["bitfield_width"] = field->getBitWidthValue(context_);
          entry["bitfield_offset"] = static_cast<int64_t>(bitOffset % 8);
        }
      }
      describeStorage(entry, field->getType());
      std::string recordId = recordSemanticId(field->getType());
      if (!recordId.empty()) entry["record_semantic_id"] = recordId;
      fields.push_back(std::move(entry));
    }

    llvm::json::Array baseOffsets;
    if (layout) {
      for (const CXXBaseSpecifier& base : record->bases()) {
        const CXXRecordDecl* baseRecord = base.getType()->getAsCXXRecordDecl();
        if (!baseRecord) continue;
        // Virtual bases use a different layout API; skip until consumers
        // understand vbtable-relative offsets (getVBaseClassOffset).
        if (base.isVirtual()) continue;
        baseOffsets.push_back(llvm::json::Object{
            {"name", typeName(base.getType())},
            {"offset", layout->getBaseClassOffset(baseRecord).getQuantity()},
        });
      }
    }

    // Every virtual introduced or overridden by this class, in declaration
    // order: the vtable order a caller's indirect call has to agree with.
    llvm::json::Array virtuals;
    for (const Decl* member : record->decls()) {
      const auto* function = dyn_cast<FunctionDecl>(member);
      if (!function || !isVirtual(function)) continue;
      virtuals.push_back(
          semanticId(function, qualify(qualifiedName, function->getNameAsString())));
    }

    llvm::json::Object payload{
        {"record", "class"},
        {"semantic_id", ("record:" + qualifiedName).str()},
        {"qualified_name", qualifiedName},
        {"bases", std::move(bases)},
        {"fields", std::move(fields)},
        {"virtual_declarations", std::move(virtuals)},
        {"source_file", relative(location.file)},
        {"line", location.line},
        {"end_line", location.endLine},
    };
    if (layout) {
      payload["size"] = layout->getSize().getQuantity();
      payload["alignment"] = layout->getAlignment().getQuantity();
      payload["base_offsets"] = std::move(baseOffsets);
    }
    emit(std::move(payload));
  }

  template <typename Predicate>
  static const Stmt* findDescendant(const Stmt* node, Predicate predicate) {
    if (!node) return nullptr;
    if (predicate(node)) return node;
    for (const Stmt* child : node->children()) {
      if (const Stmt* found = findDescendant(child, predicate)) return found;
    }
    return nullptr;
  }

  // `static_assert(sizeof(T) == N)` is the one place the recovered sources state
  // a proven layout size, so it is indexed as an assertion about a class rather
  // than left inside a function-free expression nobody reads.
  //
  // Parentheses are stepped over. Most of the recovered layout proofs are
  // written `static_assert((sizeof(T) == N), ...)`, and reading the comparison
  // through the AST-JSON shape missed every one of them, because the parentheses
  // put a node between the assertion and its comparison.
  void emitSizeAssertion(const StaticAssertDecl* assertion, llvm::StringRef scope) {
    const auto* comparison = dyn_cast<BinaryOperator>(assertion->getAssertExpr()->IgnoreParens());
    if (!comparison || comparison->getOpcode() != BO_EQ) return;
    const Stmt* sizeOf = findDescendant(comparison, [](const Stmt* node) {
      const auto* trait = dyn_cast<UnaryExprOrTypeTraitExpr>(node);
      return trait && trait->getKind() == UETT_SizeOf && trait->isArgumentType();
    });
    const Stmt* literal = findDescendant(
        comparison, [](const Stmt* node) { return isa<IntegerLiteral>(node); });
    if (!sizeOf || !literal) return;

    std::string name = cast<UnaryExprOrTypeTraitExpr>(sizeOf)->getArgumentType().getAsString(policy_);
    if (name.find("::") == std::string::npos && !scope.empty()) name = qualify(scope, name);
    emit(llvm::json::Object{
        {"record", "size-assertion"},
        {"qualified_name", name},
        {"asserted_size", cast<IntegerLiteral>(literal)->getValue().getZExtValue()},
    });
  }

  void emit(llvm::json::Object record) {
    ScopedTimer timer(profile_.serializeMs);
    std::string kind = record.getString("record").value_or("").str();
    std::string line;
    llvm::raw_string_ostream rendered(line);
    rendered << llvm::json::Value(std::move(record)) << "\n";
    rendered.flush();
    profile_.records[kind] += 1;
    profile_.bytes[kind] += static_cast<int64_t>(line.size());
    out_ << line;
  }

  // -- marker blocks ---------------------------------------------------------
  //
  // A marker block is a run of `//` comments on consecutive lines, at least one
  // of which is shaped like a reccmp marker. The marker grammar stays in reccmp;
  // the indexer only states where each block is and which declarations begin
  // at the first code token after it, so reccmp never has to find a declaration
  // by reading C++ itself.

  // Declarations a marker may annotate, keyed by the file offset where their
  // source (or an enclosing template header / `extern "C"`) begins.
  void registerAnchor(const Decl* declaration, SourceLocation begin) {
    SourceLocation location = sources_.getExpansionLoc(begin);
    if (!location.isValid() || !location.isFileID()) return;
    if (!fileInfo(location).indexed) return;
    auto [file, offset] = sources_.getDecomposedLoc(location);
    std::vector<const Decl*>& slot = anchors_[file][offset];
    if (std::find(slot.begin(), slot.end(), declaration) == slot.end()) {
      slot.push_back(declaration);
    }
  }

  static bool isAnchorKind(const Decl* declaration) {
    if (const auto* function = dyn_cast<FunctionDecl>(declaration)) {
      return !function->isImplicit() && !function->getNameAsString().empty();
    }
    if (const auto* variable = dyn_cast<VarDecl>(declaration)) {
      return !variable->isImplicit() && !isa<ParmVarDecl>(variable) &&
             !variable->getNameAsString().empty();
    }
    if (const auto* record = dyn_cast<CXXRecordDecl>(declaration)) {
      return !record->isImplicit() && record->getIdentifier();
    }
    return false;
  }

  void registerAnchors(const Decl* declaration, SourceLocation outerBegin) {
    if (!isAnchorKind(declaration)) return;
    registerAnchor(declaration, declaration->getBeginLoc());
    if (const auto* declarator = dyn_cast<DeclaratorDecl>(declaration)) {
      registerAnchor(declaration, declarator->getOuterLocStart());
    }
    if (outerBegin.isValid()) registerAnchor(declaration, outerBegin);
  }

  llvm::json::Value anchorCandidate(const Decl* declaration) const {
    if (const auto* function = dyn_cast<FunctionDecl>(declaration)) {
      std::string qualifiedName =
          qualify(scopeOf(function->getDeclContext()), function->getNameAsString());
      Location location = locate(function);
      return llvm::json::Object{
          {"kind", "function"},
          {"semantic_id", semanticId(function, qualifiedName)},
          {"qualified_name", qualifiedName},
          {"is_definition", isEmittedDefinition(function)},
          {"line", location.line},
          {"end_line", location.endLine},
      };
    }
    if (const auto* variable = dyn_cast<VarDecl>(declaration)) {
      std::string qualifiedName =
          qualify(scopeOf(variable->getDeclContext()), variable->getNameAsString());
      llvm::json::Object candidate{
          {"kind", "variable"},
          {"semantic_id", variableSemanticId(variable, qualifiedName)},
          {"qualified_name", qualifiedName},
          {"name", variable->getNameAsString()},
          {"local_static", variable->isStaticLocal()},
          {"enclosing_function", nullptr},
      };
      if (const auto* function =
              dyn_cast_or_null<FunctionDecl>(variable->getParentFunctionOrMethod())) {
        std::string functionName =
            qualify(scopeOf(function->getDeclContext()), function->getNameAsString());
        candidate["enclosing_function"] = semanticId(function, functionName);
      }
      return candidate;
    }
    const auto* record = cast<CXXRecordDecl>(declaration);
    std::string qualifiedName = qualify(scopeOf(record->getDeclContext()), component(record));
    return llvm::json::Object{
        {"kind", "class"},
        {"semantic_id", "record:" + qualifiedName},
        {"qualified_name", qualifiedName},
    };
  }

  // The first code token after `offset`, skipping comments, whitespace and
  // whole preprocessor directive lines.
  Token nextCodeToken(FileID file, unsigned offset) const {
    llvm::StringRef buffer = sources_.getBufferData(file);
    Lexer lexer(sources_.getLocForStartOfFile(file), context_.getLangOpts(), buffer.begin(),
                buffer.begin() + offset, buffer.end());
    Token token;
    lexer.LexFromRawLexer(token);
    while (token.is(tok::hash) && token.isAtStartOfLine()) {
      do {
        lexer.LexFromRawLexer(token);
      } while (!token.is(tok::eof) && !token.isAtStartOfLine());
    }
    return token;
  }

  // The first string literal (with adjacent literals concatenated) on the
  // line that starts at `token`, as the bytes the compiler would emit.
  llvm::json::Value lineString(FileID file, const Token& first) const {
    llvm::StringRef buffer = sources_.getBufferData(file);
    unsigned offset = sources_.getFileOffset(first.getLocation());
    Lexer lexer(sources_.getLocForStartOfFile(file), context_.getLangOpts(), buffer.begin(),
                buffer.begin() + offset, buffer.end());
    std::vector<Token> literal;
    Token token;
    for (bool atStart = true;; atStart = false) {
      lexer.LexFromRawLexer(token);
      if (token.is(tok::eof) || (!atStart && token.isAtStartOfLine() && literal.empty())) break;
      if (tok::isStringLiteral(token.getKind())) {
        literal.push_back(token);
      } else if (!literal.empty()) {
        break;
      }
    }
    if (literal.empty()) return nullptr;
    StringLiteralParser parser(literal, preprocessor_);
    if (parser.hadError) return nullptr;
    const TargetInfo& target = context_.getTargetInfo();
    unsigned width = 1;
    if (parser.isWide()) width = target.getWCharWidth() / 8;
    if (parser.isUTF16()) width = target.getChar16Width() / 8;
    if (parser.isUTF32()) width = target.getChar32Width() / 8;
    return llvm::json::Object{
        {"hex", llvm::toHex(parser.GetString(), /*LowerCase=*/true)},
        {"char_width", static_cast<int64_t>(width)},
    };
  }

  static bool looksLikeMarker(llvm::StringRef text) {
    static const llvm::Regex pattern(
        "^//[[:space:]]*[[:alnum:]_]+:[[:space:]]*[[:alnum:]_]+[[:space:]]+0[xX][[:xdigit:]]+");
    return pattern.match(text);
  }

  void emitMarkerBlock(FileID file, const std::vector<LineComment>& group) {
    llvm::json::Array comments;
    for (const LineComment& comment : group) {
      comments.push_back(llvm::json::Object{
          {"text", comment.text},
          {"line", comment.line},
          {"column", comment.column},
          {"offset", comment.offset},
      });
    }
    const CachedFile& info = fileInfo(sources_.getLocForStartOfFile(file));
    llvm::json::Object record{
        {"record", "marker-block"},
        {"source_file", relative(info.absolute)},
        {"comments", std::move(comments)},
        {"anchor", nullptr},
    };
    Token token = nextCodeToken(file, group.back().endOffset);
    if (!token.is(tok::eof)) {
      unsigned offset = sources_.getFileOffset(token.getLocation());
      llvm::json::Array candidates;
      auto fileAnchors = anchors_.find(file);
      if (fileAnchors != anchors_.end()) {
        const auto& byOffset = fileAnchors->second;
        auto exact = byOffset.find(offset);
        if (exact != byOffset.end()) {
          for (const Decl* declaration : exact->second) {
            candidates.push_back(anchorCandidate(declaration));
          }
        } else {
          // A macro that expands to nothing (an export decoration) may come
          // before the declaration on the same line.
          unsigned line = sources_.getLineNumber(file, offset);
          auto after = byOffset.lower_bound(offset);
          if (after != byOffset.end() && sources_.getLineNumber(file, after->first) == line) {
            for (const Decl* declaration : after->second) {
              candidates.push_back(anchorCandidate(declaration));
            }
          }
        }
      }
      record["anchor"] = llvm::json::Object{
          {"line", sources_.getLineNumber(file, offset)},
          {"column", sources_.getColumnNumber(file, offset)},
          {"candidates", std::move(candidates)},
          {"string", lineString(file, token)},
      };
    }
    emit(std::move(record));
  }

  void emitMarkerBlocks() {
    for (const auto& entry : comments_.comments()) {
      FileID file = entry.first;
      const std::vector<LineComment>& comments = entry.second;
      if (!fileInfo(sources_.getLocForStartOfFile(file)).indexed) continue;
      std::vector<LineComment> group;
      bool marked = false;
      auto flush = [&]() {
        if (marked) emitMarkerBlock(file, group);
        group.clear();
        marked = false;
      };
      for (const LineComment& comment : comments) {
        if (!group.empty() && comment.line != group.back().line + 1) flush();
        group.push_back(comment);
        marked = marked || looksLikeMarker(comment.text);
      }
      flush();
    }
  }

  void walkContext(const DeclContext* context, const std::string& scope,
                   SourceLocation outerBegin = {}) {
    for (const Decl* declaration : context->decls()) walkDecl(declaration, scope, outerBegin);
  }

  // `outerBegin` is where an enclosing template header or brace-less
  // `extern "C"` starts: a marker above either annotates the declaration.
  void walkDecl(const Decl* declaration, const std::string& scope,
                SourceLocation outerBegin = {}) {
    if (!visited_.insert(declaration).second) return;
    registerAnchors(declaration, outerBegin);

    std::string childScope = scope;
    std::string part = component(declaration);
    if (!part.empty()) childScope = qualify(scope, part);

    SourceLocation beginLoc =
        sources_.getExpansionLoc(declaration->getSourceRange().getBegin());
    const CachedFile& file = fileInfo(beginLoc);
    bool indexed = file.indexed;
    Location location;
    if (indexed) {
      location.file = file.absolute;
      PresumedLoc begin = sources_.getPresumedLoc(beginLoc);
      if (begin.isValid()) location.line = begin.getLine();
      PresumedLoc end =
          sources_.getPresumedLoc(sources_.getExpansionLoc(declaration->getSourceRange().getEnd()));
      location.endLine = end.isValid() ? end.getLine() : location.line;
    }

    if (const auto* record = dyn_cast<CXXRecordDecl>(declaration)) {
      if (indexed && record->isCompleteDefinition() && record->getIdentifier()) {
        emitClass(record, childScope, location);
      }
    } else if (const auto* function = dyn_cast<FunctionDecl>(declaration)) {
      if (indexed && !function->isImplicit() && !function->getNameAsString().empty()) {
        emitDeclaration(function, location);
      }
    } else if (const auto* variable = dyn_cast<VarDecl>(declaration)) {
      // Function-local variables have no cross-TU identity: an automatic
      // `int x` in one function and a `UINT32 x` in another share the bare
      // `_x` spelling with no linkage, and must never meet in a consistency
      // gate. Static locals are scoped to their function the same way.
      if (indexed && !variable->isImplicit() && !isa<ParmVarDecl>(variable) &&
          !variable->getDeclContext()->isFunctionOrMethod() &&
          !variable->getNameAsString().empty()) {
        emitVariable(variable, location);
      }
    } else if (const auto* assertion = dyn_cast<StaticAssertDecl>(declaration)) {
      if (indexed) emitSizeAssertion(assertion, scope);
    }

    // A template's specializations are reached through the template, exactly as
    // the AST dump reaches them: they are not members of any DeclContext, and
    // an implicit instantiation is where a recovered template body's emitted
    // code actually lives.
    if (const auto* classTemplate = dyn_cast<ClassTemplateDecl>(declaration)) {
      walkDecl(classTemplate->getTemplatedDecl(), scope, classTemplate->getBeginLoc());
      for (const auto* specialization : classTemplate->specializations()) {
        walkDecl(specialization, scope);
      }
      return;
    }
    if (const auto* functionTemplate = dyn_cast<FunctionTemplateDecl>(declaration)) {
      walkDecl(functionTemplate->getTemplatedDecl(), scope, functionTemplate->getBeginLoc());
      for (const auto* specialization : functionTemplate->specializations()) {
        walkDecl(specialization, scope);
      }
      return;
    }
    if (const auto* linkage = dyn_cast<LinkageSpecDecl>(declaration)) {
      SourceLocation outer = linkage->hasBraces() ? SourceLocation() : linkage->getBeginLoc();
      walkContext(linkage, childScope, outer);
      return;
    }

    if (const auto* inner = dyn_cast<DeclContext>(declaration)) walkContext(inner, childScope);
  }

  ASTContext& context_;
  SourceManager& sources_;
  PrintingPolicy policy_;
  mutable ASTNameGenerator names_;
  llvm::raw_ostream& out_;
  Preprocessor& preprocessor_;
  const LineCommentCollector& comments_;
  llvm::DenseSet<const Decl*> visited_;
  mutable llvm::DenseMap<FileID, CachedFile> files_;
  llvm::DenseMap<FileID, std::map<unsigned, std::vector<const Decl*>>> anchors_;
};

class IndexConsumer : public ASTConsumer {
 public:
  IndexConsumer(CompilerInstance& instance, llvm::raw_ostream& out, Profile& profile)
      : instance_(instance), out_(out), profile_(profile) {
    instance_.getPreprocessor().addCommentHandler(&comments_);
  }

  ~IndexConsumer() override { instance_.getPreprocessor().removeCommentHandler(&comments_); }

  void HandleTranslationUnit(ASTContext& context) override {
    ScopedTimer timer(profile_.consumerMs);
    const TargetInfo& target = context.getTargetInfo();
    out_ << llvm::json::Value(llvm::json::Object{
                {"record", "unit-abi"},
                {"target_triple", target.getTriple().str()},
                {"pointer_width", static_cast<int64_t>(target.getPointerWidth(LangAS::Default) / 8)},
                {"ms_abi", target.getCXXABI().isMicrosoft()},
            })
         << "\n";
    Indexer(context, out_, instance_.getPreprocessor(), comments_, profile_).run();
    // The translation unit's transitive include set is the dependency list a
    // per-unit cache needs. The preprocessor tracks it independently of any
    // DetailedRecord / PreprocessingRecord.
    llvm::json::Array dependencies;
    for (const FileEntry* file : instance_.getPreprocessor().getIncludedFiles()) {
      const llvm::StringRef path = file->tryGetRealPathName();
      if (!path.empty()) dependencies.push_back(path.str());
    }
    out_ << llvm::json::Value(llvm::json::Object{
                {"record", "dependency"},
                {"files", std::move(dependencies)},
            })
         << "\n";
    out_.flush();
  }

 private:
  CompilerInstance& instance_;
  llvm::raw_ostream& out_;
  Profile& profile_;
  LineCommentCollector comments_;
};

class IndexAction : public ASTFrontendAction {
 public:
  IndexAction(llvm::raw_ostream& out, Profile& profile) : out_(out), profile_(profile) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& instance,
                                                 llvm::StringRef) override {
    return std::make_unique<IndexConsumer>(instance, out_, profile_);
  }

 private:
  llvm::raw_ostream& out_;
  Profile& profile_;
};

class ChangeDirectory {
 public:
  explicit ChangeDirectory(llvm::StringRef directory) {
    if (std::error_code ec = llvm::sys::fs::current_path(previous_)) {
      error_ = ec;
      return;
    }
    if (std::error_code ec = llvm::sys::fs::set_current_path(directory)) {
      error_ = ec;
      previous_.clear();
      return;
    }
    active_ = true;
  }

  ~ChangeDirectory() {
    if (active_) llvm::sys::fs::set_current_path(previous_);
  }

  std::error_code error() const { return error_; }

 private:
  llvm::SmallString<256> previous_;
  std::error_code error_;
  bool active_ = false;
};

// Index one clang-cl driver command line, writing NDJSON records to `out`.
// Diagnostics go to `diagnostics`. LLVM targets must already be initialized.
int indexOneTranslationUnit(llvm::ArrayRef<const char*> argv, llvm::raw_ostream& out,
                            llvm::raw_ostream& diagnostics) {
  Profile profile;
  Clock::time_point start = Clock::now();
  if (argv.empty()) {
    diagnostics << "indexer: empty driver command line\n";
    return 1;
  }

  // The compile database records clang-cl command lines, so the driver has to
  // select cl mode the same way the real build selects it - from the program
  // name - before it parses anything else. Passing the mode the name implies as
  // an option states it where the driver cannot mistake it.
  driver::ParsedClangName parsedName =
      driver::ToolChain::getTargetAndModeFromProgramName(argv[0]);
  std::vector<const char*> arguments{argv[0]};
  if (parsedName.DriverMode) arguments.push_back(parsedName.DriverMode);
  arguments.insert(arguments.end(), argv.begin() + 1, argv.end());

  llvm::IntrusiveRefCntPtr<DiagnosticOptions> diagnosticOptions(new DiagnosticOptions());
  TextDiagnosticPrinter printer(diagnostics, diagnosticOptions.get());
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> diagnosticIds(new DiagnosticIDs());
  DiagnosticsEngine engine(diagnosticIds, diagnosticOptions, &printer,
                           /*ShouldOwnClient=*/false);

  // The resource directory comes from the resolved executable, which is what
  // clang's own main does: /usr/bin/clang-cl is a symlink, and the builtin
  // headers sit next to its target.
  llvm::SmallString<128> executable(arguments[0]);
  llvm::sys::fs::real_path(arguments[0], executable, /*expand_tilde=*/false);
  driver::Driver theDriver(executable, llvm::sys::getDefaultTargetTriple(), engine);
  theDriver.setTargetAndMode(parsedName);
  theDriver.setCheckInputsExist(false);

  std::unique_ptr<driver::Compilation> compilation(theDriver.BuildCompilation(arguments));
  if (!compilation || engine.hasErrorOccurred()) {
    diagnostics << "indexer: the driver rejected the command line\n";
    return 1;
  }
  const driver::Command* compile = nullptr;
  for (const driver::Command& command : compilation->getJobs()) {
    if (!command.getArguments().empty() &&
        llvm::StringRef(command.getArguments().front()) == "-cc1") {
      compile = &command;
      break;
    }
  }
  if (!compile) {
    diagnostics << "indexer: the command line produced no compilation job\n";
    return 1;
  }

  auto invocation = std::make_shared<CompilerInvocation>();
  if (!CompilerInvocation::CreateFromArgs(*invocation, compile->getArguments(), engine)) {
    return 1;
  }
  CompilerInstance instance;
  instance.setInvocation(std::move(invocation));
  instance.createDiagnostics(&printer, /*ShouldOwnClient=*/false);
  if (!instance.hasDiagnostics()) return 1;
  profile.invocationMs = millisecondsSince(start);

  IndexAction action(out, profile);
  Clock::time_point frontend = Clock::now();
  if (!instance.ExecuteAction(action)) return 1;
  profile.frontendMs = millisecondsSince(frontend);
  if (instance.getDiagnostics().hasErrorOccurred()) return 1;
  out << llvm::json::Value(profile.toJson()) << "\n";
  return 0;
}

// Index one job: {"directory", "output", "arguments"}. Returns the reply
// sent back to reccmp: {"output", "ok", "diagnostics"}.
llvm::json::Object runJob(const llvm::json::Object& job) {
  std::optional<llvm::StringRef> directory = job.getString("directory");
  std::optional<llvm::StringRef> output = job.getString("output");
  const llvm::json::Array* arguments = job.getArray("arguments");
  llvm::json::Object reply{{"output", output ? output->str() : ""}, {"ok", false}};
  if (!directory || !output || !arguments || arguments->empty()) {
    reply["diagnostics"] = "job needs directory, output, and arguments";
    return reply;
  }
  std::vector<std::string> storage;
  for (const llvm::json::Value& value : *arguments) {
    std::optional<llvm::StringRef> argument = value.getAsString();
    if (!argument) {
      reply["diagnostics"] = "arguments must be strings";
      return reply;
    }
    storage.emplace_back(argument->str());
  }
  std::vector<const char*> argv;
  for (const std::string& argument : storage) argv.push_back(argument.c_str());

  ChangeDirectory cwd(*directory);
  if (cwd.error()) {
    reply["diagnostics"] = ("cannot chdir to " + *directory + ": " + cwd.error().message()).str();
    return reply;
  }
  std::error_code ec;
  llvm::raw_fd_ostream fileOut(*output, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    reply["diagnostics"] = ("cannot write " + *output + ": " + ec.message()).str();
    return reply;
  }
  std::string diagnosticText;
  llvm::raw_string_ostream diagnosticStream(diagnosticText);
  int status = indexOneTranslationUnit(argv, fileOut, diagnosticStream);
  fileOut.close();
  diagnosticStream.flush();
  if (status != 0) llvm::sys::fs::remove(*output);
  reply["ok"] = status == 0;
  reply["diagnostics"] = diagnosticText;
  return reply;
}

// A persistent worker: one JSON job per stdin line, one JSON reply per stdout
// line. LLVM initialization happens once per worker, and reccmp hands each
// idle worker the next job, so a slow unit never holds up a whole chunk.
int serve() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (llvm::StringRef(line).trim().empty()) continue;
    llvm::json::Object reply;
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(line);
    if (!parsed) {
      reply = llvm::json::Object{{"output", ""}, {"ok", false},
                                 {"diagnostics", llvm::toString(parsed.takeError())}};
    } else if (const llvm::json::Object* job = parsed->getAsObject()) {
      reply = runJob(*job);
    } else {
      reply = llvm::json::Object{{"output", ""}, {"ok", false},
                                 {"diagnostics", "job is not a JSON object"}};
    }
    llvm::outs() << llvm::json::Value(std::move(reply)) << "\n";
    llvm::outs().flush();
  }
  return 0;
}

}  // namespace

int main(int argc, const char** argv) {
  // The identity of the Clang libraries this collector runs against, which
  // decide its output as much as its own source does.
  if (argc == 2 && llvm::StringRef(argv[1]) == "--version") {
    llvm::outs() << clang::getClangFullVersion() << "\n";
    return 0;
  }
  const char* root = std::getenv("RECCMP_SOURCE_ROOT");
  if (!root) {
    llvm::errs() << "RECCMP_SOURCE_ROOT is required\n";
    return 2;
  }
  repositoryPrefix = root;
  if (repositoryPrefix.back() != '/') repositoryPrefix += '/';
  kRepositoryPrefix = repositoryPrefix;

  // The VC6 headers contain MS-style inline assembly, which Sema refuses to
  // accept unless the target's assembly parser is registered. Done once so
  // batch workers do not repeat it per translation unit.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();

  if (argc == 2 && llvm::StringRef(argv[1]) == "--serve") return serve();
  if (argc < 3) {
    llvm::errs() << "usage: indexer <clang-cl driver command line...>\n"
                 << "       indexer --serve   (jobs on stdin, replies on stdout)\n"
                 << "       indexer --version\n";
    return 2;
  }
  return indexOneTranslationUnit(llvm::ArrayRef(argv + 1, argv + argc), llvm::outs(),
                                 llvm::errs());
}
