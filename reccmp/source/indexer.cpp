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
#include "llvm/Support/VirtualFileSystem.h"
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

  static const char* storageKind(QualType type) {
    QualType current = type.getCanonicalType();
    if (current->getAs<ReferenceType>()) return "reference";
    if (current->getAs<PointerType>() || pointerDepth(type) > 0) return "pointer";
    if (current->getAsArrayTypeUnsafe()) return "array";
    if (current->getAsCXXRecordDecl()) return "embedded_record";
    return "scalar";
  }

  void describeStorage(llvm::json::Object& entry, QualType type) const {
    const char* kind = storageKind(type);
    entry["storage_kind"] = kind;
    if (const ArrayType* array = type.getCanonicalType()->getAsArrayTypeUnsafe()) {
      QualType element = array->getElementType();
      entry["array_element_type"] = typeName(element);
      if (!element->isDependentType() && !element->isIncompleteType() &&
          element->isConstantSizeType()) {
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
    }
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

  std::vector<std::string> parameterTypes(const FunctionDecl* function) const {
    std::vector<std::string> parameters;
    for (const ParmVarDecl* parameter : function->parameters()) {
      parameters.push_back(canonicalName(parameter->getType()));
    }
    return parameters;
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

    // The compared identity dissolves typedef and elaborated spellings.
    std::string returnType;
    if (semanticKind != "constructor" && semanticKind != "destructor") {
      returnType = canonicalName(function->getReturnType());
    }
    llvm::json::Array parameters;
    for (const std::string& parameter : parameterTypes(function)) parameters.push_back(parameter);

    // The signature fields feed the cross-unit declaration consistency gate.
    llvm::json::Object record{
        {"record", "declaration"},
        {"semantic_id", functionIdentity},
        {"qualified_name", qualifiedName},
        {"semantic_kind", semanticKind},
        {"calling_convention", callingConvention(function)},
        {"return_type", returnType},
        {"parameter_types", std::move(parameters)},
        {"is_variadic", function->isVariadic()},
        {"linkage", linkageName(function->getLinkageInternal())},
        {"storage_class", storageClassName(function->getStorageClass())},
        {"owning_class", isMember ? llvm::json::Value(scope) : llvm::json::Value(nullptr)},
        {"source_file", relative(location.file)},
        {"line", location.line},
        {"end_line", location.endLine},
        // A primary template's definition is not a concrete emitted function
        // that owns a reccmp FUNCTION marker. Keep its marker-join row
        // declaration-only; an emitted specialization carries the concrete
        // function identity and extent.
        {"is_definition", isEmittedDefinition(function)},
    };
    emit(std::move(record));
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
    payload["storage_kind"] = storageKind(variable->getType());
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
          entry["bitfield_width"] = field->getBitWidthValue();
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
            {"semantic_id", recordSemanticId(base.getType())},
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

  void emitSizeAssertion(const StaticAssertDecl* assertion) {
    const auto* comparison =
        dyn_cast<BinaryOperator>(assertion->getAssertExpr()->IgnoreParenImpCasts());
    if (!comparison || comparison->getOpcode() != BO_EQ) return;
    const Expr* left = comparison->getLHS()->IgnoreParenImpCasts();
    const Expr* right = comparison->getRHS()->IgnoreParenImpCasts();
    const auto* sizeOf = dyn_cast<UnaryExprOrTypeTraitExpr>(left);
    const auto* literal = dyn_cast<IntegerLiteral>(right);
    if (!sizeOf || !literal) {
      sizeOf = dyn_cast<UnaryExprOrTypeTraitExpr>(right);
      literal = dyn_cast<IntegerLiteral>(left);
    }
    if (!sizeOf || sizeOf->getKind() != UETT_SizeOf ||
        !sizeOf->isArgumentType() || !literal) return;

    const CXXRecordDecl* record =
        sizeOf->getArgumentType()->getAsCXXRecordDecl();
    if (!record) return;
    std::string name = qualify(scopeOf(record->getDeclContext()), component(record));
    emit(llvm::json::Object{
        {"record", "size-assertion"},
        {"qualified_name", name},
        {"asserted_size", literal->getValue().getZExtValue()},
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

  // The `#` of a `#define` directive that directly follows `offset`, past
  // comments and whitespace only: a string written once under a name, which
  // a STRING marker above it annotates.
  bool definitionAfter(FileID file, unsigned offset, Token& hash) const {
    llvm::StringRef buffer = sources_.getBufferData(file);
    Lexer lexer(sources_.getLocForStartOfFile(file), context_.getLangOpts(), buffer.begin(),
                buffer.begin() + offset, buffer.end());
    lexer.LexFromRawLexer(hash);
    if (!hash.is(tok::hash) || !hash.isAtStartOfLine()) return false;
    Token keyword;
    lexer.LexFromRawLexer(keyword);
    return keyword.is(tok::raw_identifier) && keyword.getRawIdentifier() == "define";
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

  // The first empty line between a marker block and the code it annotates:
  // block comments or preprocessor lines in between are not blank.
  llvm::json::Value blankLineBetween(FileID file, const LineComment& last,
                                     unsigned anchorOffset) const {
    llvm::StringRef buffer = sources_.getBufferData(file);
    if (anchorOffset <= last.endOffset || anchorOffset > buffer.size()) return nullptr;
    // The slice starts at the end of the block's last line (its range
    // excludes the newline) and ends where the anchor's line begins it.
    llvm::SmallVector<llvm::StringRef, 8> lines;
    buffer.slice(last.endOffset, anchorOffset).split(lines, '\n');
    for (size_t index = 1; index + 1 < lines.size(); ++index) {
      if (lines[index].trim().empty()) return static_cast<int64_t>(last.line + index);
    }
    return nullptr;
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
    Token definition;
    if (definitionAfter(file, group.back().endOffset, definition)) {
      unsigned offset = sources_.getFileOffset(definition.getLocation());
      record["anchor"] = llvm::json::Object{
          {"line", sources_.getLineNumber(file, offset)},
          {"column", sources_.getColumnNumber(file, offset)},
          {"candidates", llvm::json::Array{}},
          {"string", lineString(file, definition)},
          {"blank_line", blankLineBetween(file, group.back(), offset)},
      };
      emit(std::move(record));
      return;
    }
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
          {"blank_line", blankLineBetween(file, group.back(), offset)},
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
    if (!part.empty()) childScope = qualify(scopeOf(declaration->getDeclContext()), part);

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
      if (indexed) emitSizeAssertion(assertion);
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

  DiagnosticOptions diagnosticOptions;
  TextDiagnosticPrinter printer(diagnostics, diagnosticOptions);
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
  CompilerInstance instance(std::move(invocation));
  instance.createDiagnostics(*llvm::vfs::getRealFileSystem(), &printer,
                             /*ShouldOwnClient=*/false);
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
