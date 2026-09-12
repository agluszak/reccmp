// Emit the source-index records for one translation unit directly from Clang's
// AST. Partitioning by link namespace / TU, winner selection, conflicts, marker
// binding, asserted sizes and vtable addresses stay in reccmp, which owns them.
//
// Single-TU mode writes NDJSON to stdout. Batch mode (`--batch <manifest.jsonl>`)
// indexes many units in one process so LLVM target initialization happens once.
//
// Output is one JSON object per line: `{"record":"declaration",...}`,
// `{"record":"variable",...}`, `{"record":"class",...}`,
// `{"record":"size-assertion",...}` or `{"record":"dependency",...}`.

#include <cstdlib>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Mangle.h"
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
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
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
};

struct CachedFile {
  bool indexed = false;
  std::string absolute;
};

class Indexer {
 public:
  Indexer(ASTContext& context, llvm::raw_ostream& out)
      : context_(context),
        sources_(context.getSourceManager()),
        policy_(context.getPrintingPolicy()),
        names_(context),
        out_(out) {}

  void run() { walkContext(context_.getTranslationUnitDecl(), ""); }

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
    Location location;
    SourceRange range = declaration->getSourceRange();
    SourceLocation beginLoc = sources_.getExpansionLoc(range.getBegin());
    const CachedFile& file = fileInfo(beginLoc);
    location.file = file.absolute;
    PresumedLoc begin = sources_.getPresumedLoc(beginLoc);
    if (begin.isValid()) location.line = begin.getLine();
    PresumedLoc end = sources_.getPresumedLoc(sources_.getExpansionLoc(range.getEnd()));
    location.endLine = end.isValid() ? end.getLine() : location.line;
    return location;
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

  static bool hasThis(llvm::StringRef semanticKind) {
    return semanticKind == "constructor" || semanticKind == "destructor" ||
           semanticKind == "instance_method";
  }

  // On the x86 MS ABI the convention is a consequence of the semantic kind
  // unless the source spells one, and a spelled convention survives into the
  // printed function type.
  static std::string callingConvention(llvm::StringRef functionType,
                                       llvm::StringRef semanticKind) {
    std::string lowered = functionType.lower();
    for (llvm::StringRef spelling :
         {"thiscall", "stdcall", "fastcall", "vectorcall", "cdecl"}) {
      if (lowered.find(spelling.str()) != std::string::npos) {
        return ("__" + spelling).str();
      }
    }
    return hasThis(semanticKind) ? "__thiscall" : "__cdecl";
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

  void emitDeclaration(const FunctionDecl* function, const Location& location) {
    const DeclContext* context = function->getDeclContext();
    bool isMember = isa<CXXRecordDecl>(context);
    std::string scope = scopeOf(context);
    std::string qualifiedName = qualify(scope, function->getNameAsString());

    std::string semanticKind;
    if (isa<CXXConstructorDecl>(function)) {
      semanticKind = "constructor";
    } else if (isa<CXXDestructorDecl>(function)) {
      semanticKind = "destructor";
    } else if (isMember) {
      semanticKind = function->getCanonicalDecl()->getStorageClass() == SC_Static
                         ? "static_method"
                         : "instance_method";
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

    std::string convention = callingConvention(functionType, semanticKind);
    llvm::json::Object record{
        {"record", "declaration"},
        {"semantic_id", semanticId(function, qualifiedName)},
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
        // clang-cl delays template body parsing, so a pattern in a unit that
        // never instantiated it has no body and no closing brace to report. Such
        // a unit does not own the definition; the unit that instantiated it does,
        // and only that one carries the real source extent.
        {"is_definition",
         function->doesThisDeclarationHaveABody() && !function->isLateTemplateParsed()},
    };
    emit(std::move(record));
  }

  // One variable definition or declaration. Parameters are VarDecls too,
  // but they never reach this emitter: the walker filters them out, along
  // with implicit declarations such as a function body's `__func__`.
  // Mangling a variable in a dependent context is meaningless, so those fall
  // back to a qualified signature identity, mirroring uninstantiated
  // template patterns for functions.
  void emitVariable(const VarDecl* variable, const Location& location) {
    const DeclContext* context = variable->getDeclContext();
    std::string scope = scopeOf(context);
    std::string qualifiedName = qualify(scope, variable->getNameAsString());
    std::string type = canonicalName(variable->getType());
    std::string mangled;
    if (!context->isDependentContext()) mangled = names_.getName(variable);
    std::string semanticId = mangled;
    if (semanticId.empty()) semanticId = "VarDecl:" + qualifiedName + "(" + type + ")";
    emit(llvm::json::Object{
        {"record", "variable"},
        {"semantic_id", semanticId},
        {"qualified_name", qualifiedName},
        {"type", type},
        {"linkage", linkageName(variable->getLinkageInternal())},
        {"storage_class", storageClassName(variable->getStorageClass())},
        {"definition_kind",
         definitionKindName(variable->isThisDeclarationADefinition())},
        {"source_file", relative(location.file)},
        {"line", location.line},
        {"end_line", location.endLine},
    });
  }

  static bool isVirtual(const FunctionDecl* function) {
    const auto* method = dyn_cast<CXXMethodDecl>(function);
    return method && method->isVirtual();
  }

  void emitClass(const CXXRecordDecl* record, llvm::StringRef qualifiedName,
                 const Location& location) {
    llvm::json::Array bases;
    for (const CXXBaseSpecifier& base : record->bases()) bases.push_back(typeName(base.getType()));

    llvm::json::Array fields;
    for (const FieldDecl* field : record->fields()) {
      if (!field->getIdentifier()) continue;
      Location where = locate(field);
      fields.push_back(llvm::json::Object{
          {"name", field->getNameAsString()},
          {"type", typeName(field->getType())},
          {"pointer_depth", pointerDepth(field->getType())},
          {"source_file", relative(where.file)},
          {"line", where.line},
      });
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

    emit(llvm::json::Object{
        {"record", "class"},
        {"semantic_id", ("record:" + qualifiedName).str()},
        {"qualified_name", qualifiedName},
        {"bases", std::move(bases)},
        {"fields", std::move(fields)},
        {"virtual_declarations", std::move(virtuals)},
        {"source_file", relative(location.file)},
        {"line", location.line},
        {"end_line", location.endLine},
    });
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
    out_ << llvm::json::Value(std::move(record)) << "\n";
  }

  void walkContext(const DeclContext* context, const std::string& scope) {
    for (const Decl* declaration : context->decls()) walkDecl(declaration, scope);
  }

  void walkDecl(const Decl* declaration, const std::string& scope) {
    if (!visited_.insert(declaration).second) return;

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
      walkDecl(classTemplate->getTemplatedDecl(), scope);
      for (const auto* specialization : classTemplate->specializations()) {
        walkDecl(specialization, scope);
      }
      return;
    }
    if (const auto* functionTemplate = dyn_cast<FunctionTemplateDecl>(declaration)) {
      walkDecl(functionTemplate->getTemplatedDecl(), scope);
      for (const auto* specialization : functionTemplate->specializations()) {
        walkDecl(specialization, scope);
      }
      return;
    }

    if (const auto* inner = dyn_cast<DeclContext>(declaration)) walkContext(inner, childScope);
  }

  ASTContext& context_;
  SourceManager& sources_;
  PrintingPolicy policy_;
  mutable ASTNameGenerator names_;
  llvm::raw_ostream& out_;
  llvm::DenseSet<const Decl*> visited_;
  mutable llvm::DenseMap<FileID, CachedFile> files_;
};

class IndexConsumer : public ASTConsumer {
 public:
  IndexConsumer(CompilerInstance& instance, llvm::raw_ostream& out)
      : instance_(instance), out_(out) {}

  void HandleTranslationUnit(ASTContext& context) override {
    Indexer(context, out_).run();
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
};

class IndexAction : public ASTFrontendAction {
 public:
  explicit IndexAction(llvm::raw_ostream& out) : out_(out) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& instance,
                                                 llvm::StringRef) override {
    return std::make_unique<IndexConsumer>(instance, out_);
  }

 private:
  llvm::raw_ostream& out_;
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

  IndexAction action(out);
  if (!instance.ExecuteAction(action)) return 1;
  return instance.getDiagnostics().hasErrorOccurred() ? 1 : 0;
}

int runBatch(llvm::StringRef manifestPath) {
  auto buffer = llvm::MemoryBuffer::getFile(manifestPath);
  if (!buffer) {
    llvm::errs() << "indexer: cannot read manifest " << manifestPath << ": "
                 << buffer.getError().message() << "\n";
    return 1;
  }

  bool failed = false;
  llvm::StringRef remaining = buffer.get()->getBuffer();
  while (!remaining.empty()) {
    auto [line, rest] = remaining.split('\n');
    remaining = rest;
    line = line.trim();
    if (line.empty()) continue;

    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(line);
    if (!parsed) {
      llvm::errs() << "indexer: bad manifest line: "
                   << llvm::toString(parsed.takeError()) << "\n";
      failed = true;
      continue;
    }
    llvm::json::Object* object = parsed->getAsObject();
    if (!object) {
      llvm::errs() << "indexer: manifest line is not a JSON object\n";
      failed = true;
      continue;
    }
    std::optional<llvm::StringRef> directory = object->getString("directory");
    std::optional<llvm::StringRef> output = object->getString("output");
    llvm::json::Array* arguments = object->getArray("arguments");
    if (!directory || !output || !arguments || arguments->empty()) {
      llvm::errs() << "indexer: manifest line needs directory, output, and arguments\n";
      failed = true;
      continue;
    }

    std::vector<std::string> storage;
    storage.reserve(arguments->size());
    for (const llvm::json::Value& value : *arguments) {
      std::optional<llvm::StringRef> argument = value.getAsString();
      if (!argument) {
        llvm::errs() << "indexer: argument list must be strings\n";
        storage.clear();
        break;
      }
      storage.emplace_back(argument->str());
    }
    if (storage.empty()) {
      failed = true;
      continue;
    }
    std::vector<const char*> argv;
    argv.reserve(storage.size());
    for (const std::string& argument : storage) argv.push_back(argument.c_str());
    llvm::StringRef mainFile = storage.back();

    ChangeDirectory cwd(*directory);
    if (cwd.error()) {
      llvm::errs() << "indexer: " << mainFile << ": cannot chdir to " << *directory << ": "
                   << cwd.error().message() << "\n";
      failed = true;
      continue;
    }

    std::error_code ec;
    llvm::raw_fd_ostream fileOut(*output, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "indexer: " << mainFile << ": cannot write " << *output << ": "
                   << ec.message() << "\n";
      failed = true;
      continue;
    }

    std::string diagnosticText;
    llvm::raw_string_ostream diagnosticStream(diagnosticText);
    int status = indexOneTranslationUnit(argv, fileOut, diagnosticStream);
    fileOut.close();
    if (status == 0) continue;

    llvm::sys::fs::remove(*output);
    llvm::errs() << "indexer: " << mainFile << ":\n" << diagnosticText;
    failed = true;
  }
  return failed ? 1 : 0;
}

}  // namespace

int main(int argc, const char** argv) {
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

  if (argc >= 2 && llvm::StringRef(argv[1]) == "--batch") {
    if (argc != 3) {
      llvm::errs() << "usage: indexer --batch <manifest.jsonl>\n";
      return 2;
    }
    return runBatch(argv[2]);
  }
  if (argc < 3) {
    llvm::errs() << "usage: indexer <clang-cl driver command line...>\n"
                 << "       indexer --batch <manifest.jsonl>\n";
    return 2;
  }
  return indexOneTranslationUnit(llvm::ArrayRef(argv + 1, argv + argc), llvm::outs(),
                                 llvm::errs());
}
