#!/usr/bin/env python3
"""FC blind relabel of the 6 F2 methods (kotlin) -> adjudication/kotlin-flow-relabel.jsonl.
Hand-labelled from sample/kotlin-flow.jsonl + the flow brief; this script only serialises."""
import json, os
O = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kotlin-flow-relabel.jsonl')
P = lambda i, n: {"point": "param", "index": i, "name": n}
C = lambda n, o, l: {"point": "call", "calleeName": n, "ordinal": o, "line": l}
R = lambda o, l: {"point": "ret", "ordinal": o, "line": l}
K = lambda l, c: {"point": "cond", "line": l, "code": c}
def d(s, t, e, c="sure"): return {"kind": "data", "source": s, "sink": t, "confidence": c, "evidence": e}
def k(s, t, e, c="sure"): return {"kind": "control", "source": s, "sink": t, "confidence": c, "evidence": e}
F = "KtorAdminLibrary/src/main/kotlin/ir/amirroid/ktoradmin/"
def M(o, p, q, pairs, unl=None):
    r = {"ordinal": o, "path": F + p, "qname": F + p + "::" + q, "pairs": pairs}
    if unl: r["unlisted"] = unl
    return r
THIS = P(-1, "this")
rows = []
# 2 badRequest: lambda body belongs to a separate symbol
rows.append(M(2, "utils/ErrorHandler.kt", "badRequest", [
  d(THIS, C("respondText", 0, 12), "implicit extension receiver is receiver of respondText, ErrorHandler.kt:12", "unsure")]))
# 4 handleAddNewItem
H = K(36, "panel == null || panel.isShowInAdminPanel().not()")
A = K(36, "panel == null")
rows.append(M(4, "modules/add/GetAddsController.kt", "handleAddNewItem", [
  d(P(0, "panels"), C("find", 0, 35), "panels.find{} GetAddsController.kt:35"),
  d(C("find", 0, 35), A, "panel = find; tested :36"),
  d(C("find", 0, 35), H, "panel read in 2nd operand of header :36"),
  d(C("find", 0, 35), C("isShowInAdminPanel", 0, 36), "panel.isShowInAdminPanel() :36"),
  d(C("find", 0, 35), C("checkHasRole", 0, 39), "checkHasRole(panel) :39"),
  d(C("isShowInAdminPanel", 0, 36), C("not", 0, 36), "receiver of .not() :36"),
  d(C("not", 0, 36), H, "operand of header :36"),
  d(THIS, C("notFound", 0, 37), "implicit receiver :37", "unsure"),
  d(THIS, C("checkHasRole", 0, 39), "implicit receiver :39", "unsure"),
  k(A, C("isShowInAdminPanel", 0, 36), "|| short-circuit :36"),
  k(A, C("not", 0, 36), "|| short-circuit, evaluated only if A false :36", "unsure"),
  k(H, C("notFound", 0, 37), "if branch :36-37"),
  k(H, C("checkHasRole", 0, 39), "else branch :38-39")],
  [{"line": 34, "construct": "indexing parameters[...] not a call site"}]))
# 12 requiredSave
c80 = K(80, "exists().not()"); c87 = K(87, "file.exists()")
rows.append(M(12, "providers/LocalStorageProvider.kt", "LocalStorageProvider.requiredSave", [
  d(THIS, C("exists", 0, 80), "implicit receiver :80", "unsure"),
  d(THIS, C("createNewFile", 0, 81), "implicit receiver :81", "unsure"),
  d(THIS, R(0, 82), "return this :82"),
  d(THIS, C("exists", 1, 87), "file = this :85 reaches file.exists() :87 on first test"),
  d(THIS, C("createNewFile", 1, 93), "file = this :85 reaches file.createNewFile() :93 if loop skipped"),
  d(THIS, R(1, 94), "file = this :85 reaches return file :94 if loop skipped"),
  d(THIS, C("File", 0, 90), "baseName/file.extension from this reach File arg :86-90", "unsure"),
  d(P(0, "path"), C("File", 0, 90), "\"$path/$filename\" :90"),
  d(C("exists", 0, 80), C("not", 0, 80), "exists().not() :80"),
  d(C("not", 0, 80), c80, "if condition :80"),
  d(C("exists", 1, 87), c87, "while header :87"),
  d(C("File", 0, 90), C("exists", 1, 87), "file = File(..) :90 loops to :87"),
  d(C("File", 0, 90), C("createNewFile", 1, 93), "file after loop :93"),
  d(C("File", 0, 90), R(1, 94), "return file :94"),
  d(C("File", 0, 90), C("File", 0, 90), "file.extension -> filename -> File(..) next iteration :88-90", "unsure"),
  k(c80, C("createNewFile", 0, 81), "if body :80-81"),
  k(c80, R(0, 82), "if body :82"),
  k(c80, c87, "early return :82 skips while", "sure"),
  k(c80, C("exists", 1, 87), "early return skips while header evaluation", "unsure"),
  k(c80, C("createNewFile", 1, 93), "early return skips :93"),
  k(c80, R(1, 94), "early return skips :94"),
  k(c87, C("File", 0, 90), "while body :87-90")]))
# 19 registerCustomPageEntry
c264 = K(264, "_customPageEntries.containsKey(entry.path)")
rows.append(M(19, "configuration/DynamicConfiguration.kt", "DynamicConfiguration.registerCustomPageEntry", [
  d(P(0, "entry"), C("containsKey", 0, 264), "entry.path argument :264"),
  d(P(0, "entry"), C("IllegalStateException", 0, 265), "${entry.path} in message :265"),
  d(C("containsKey", 0, 264), c264, "if condition :264"),
  k(c264, C("IllegalStateException", 0, 265), "if body :264-265")],
  [{"line": 267, "construct": "indexed assignment set operator, not a call site"}]))
# 28 generatePdf
cj = K(57, "is AdminJdbcTable"); cm = K(58, "is AdminMongoCollection")
pr = []
for i, n in enumerate(["panel", "primaryKey", "call", "regularFontPath", "boldFontPath"]):
    pr.append(d(P(i, n), C("generateJdbcPdf", 0, 57), f"{n} argument/receiver :57"))
    pr.append(d(P(i, n), C("generateMongoPdf", 0, 58), f"{n} argument/receiver :58"))
pr += [d(P(0, "panel"), cj, "when subject tested :56-57", "unsure"),
       d(P(0, "panel"), cm, "when subject tested :56-58", "unsure"),
       d(C("generateJdbcPdf", 0, 57), R(0, 56), "branch value is the expression body result :56-57"),
       d(C("generateMongoPdf", 0, 58), R(0, 56), "branch value is the expression body result :56-58"),
       k(cj, C("generateJdbcPdf", 0, 57), "when branch :57"),
       k(cj, cm, "next branch condition decided by nearest before :57-58"),
       k(cm, C("generateMongoPdf", 0, 58), "when branch :58")]
rows.append(M(28, "pdf/PdfHelper.kt", "PdfHelper.generatePdf", pr))
# 29 getPrimaryKeyFilter
h = K(206, 'getPrimaryKeyField().type == FieldType.ObjectId || getPrimaryKey() == "_id"')
a = K(206, "getPrimaryKeyField().type == FieldType.ObjectId")
rows.append(M(29, "repository/MongoClientRepository.kt", "MongoClientRepository.getPrimaryKeyFilter", [
  d(THIS, C("getPrimaryKeyField", 0, 206), "implicit extension receiver :206", "unsure"),
  d(THIS, C("getPrimaryKey", 0, 206), "implicit extension receiver :206", "unsure"),
  d(THIS, C("getPrimaryKey", 1, 207), "implicit extension receiver :207", "unsure"),
  d(THIS, C("getPrimaryKey", 2, 209), "implicit extension receiver :209", "unsure"),
  d(P(0, "primaryKey"), C("ObjectId", 0, 207), "ObjectId(primaryKey) :207"),
  d(P(0, "primaryKey"), C("eq", 1, 209), "eq(.., primaryKey) :209"),
  d(C("getPrimaryKeyField", 0, 206), a, "getPrimaryKeyField().type == .. :206"),
  d(C("getPrimaryKey", 0, 206), h, "getPrimaryKey() == \"_id\" in header :206"),
  d(C("getPrimaryKey", 1, 207), C("eq", 0, 207), "eq(getPrimaryKey(), ..) :207"),
  d(C("ObjectId", 0, 207), C("eq", 0, 207), "eq(.., ObjectId(..)) :207"),
  d(C("getPrimaryKey", 2, 209), C("eq", 1, 209), "eq(getPrimaryKey(), ..) :209"),
  d(C("eq", 0, 207), R(0, 206), "if-expression value is the body :206-207"),
  d(C("eq", 1, 209), R(0, 206), "else value is the body :208-209"),
  k(a, C("getPrimaryKey", 0, 206), "|| short-circuit :206"),
  k(h, C("getPrimaryKey", 1, 207), "if branch :206-207"),
  k(h, C("ObjectId", 0, 207), "if branch :207"),
  k(h, C("eq", 0, 207), "if branch :207"),
  k(h, C("getPrimaryKey", 2, 209), "else branch :208-209"),
  k(h, C("eq", 1, 209), "else branch :209")]))
with open(O, 'w') as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(len(rows), sum(len(r['pairs']) for r in rows))
