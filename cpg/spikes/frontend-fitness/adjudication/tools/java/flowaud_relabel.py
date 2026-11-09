#!/usr/bin/env python3
"""Blind relabel of the F2 methods, written from sample/java-flow.jsonl + brief only."""
import json

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
FF = (_CPG+"/spikes/frontend-fitness")
def P(i, n): return {"point": "param", "index": i, "name": n}
def C(n, o, l): return {"point": "call", "calleeName": n, "ordinal": o, "line": l}
def R(o, l): return {"point": "ret", "ordinal": o, "line": l}
def K(l, c): return {"point": "cond", "line": l, "code": c}
def build(ordinal, path, pairs):
    out = []
    for kind, s, t, ev in pairs:
        out.append({"kind": kind, "source": s, "sink": t, "confidence": "sure", "evidence": ev[:300]})
    return {"ordinal": ordinal, "path": path, "qname": f"{path}::" , "pairs": out}
JD = "src/main/java/org/springframework/samples/petclinic/repository/jdbc/"
RC = "src/main/java/org/springframework/samples/petclinic/rest/controller/v1/"
M = {}
# ---- 4 JdbcOwnerRepositoryImpl.delete
p = JD + "JdbcOwnerRepositoryImpl.java"; E = "JdbcOwnerRepositoryImpl.java:"
c1 = K(220, "pet : pets"); c2 = K(225, "visit : visits")
M[4] = (p, p + "::JdbcOwnerRepositoryImpl.delete", [
 ("data", P(0,"owner"), C("getId",0,217), E+"217"),
 ("data", P(0,"owner"), C("getPets",0,218), E+"218"),
 ("data", C("HashMap",0,216), C("put",0,217), E+"216-217 receiver"),
 ("data", C("HashMap",0,216), C("update",2,232), E+"216,232 ownerParams arg"),
 ("data", C("getId",0,217), C("put",0,217), E+"217"),
 ("data", C("getPets",0,218), c1, E+"218,220"),
 ("data", C("getPets",0,218), C("getId",1,222), E+"pet loop var 220,222"),
 ("data", C("getPets",0,218), C("getVisits",0,224), E+"pet loop var 220,224"),
 ("data", C("HashMap",1,221), C("put",1,222), E+"221-222"),
 ("data", C("HashMap",1,221), C("update",1,230), E+"221,230 petParams"),
 ("data", C("getId",1,222), C("put",1,222), E+"222"),
 ("data", C("getVisits",0,224), c2, E+"224-225"),
 ("data", C("getVisits",0,224), C("getId",2,227), E+"visit loop var 225,227"),
 ("data", C("HashMap",2,226), C("put",2,227), E+"226-227"),
 ("data", C("HashMap",2,226), C("update",0,228), E+"226,228 visitParams"),
 ("data", C("getId",2,227), C("put",2,227), E+"227"),
 ("control", c1, c2, E+"220-225 nested loop"),
 ] + [("control", c1, C(n,o,l), E+f"loop body 220-231, {n}@{l}") for n,o,l in
      [("HashMap",1,221),("put",1,222),("getId",1,222),("getVisits",0,224),("update",1,230)]]
   + [("control", c2, C(n,o,l), E+f"loop body 225-229, {n}@{l}") for n,o,l in
      [("HashMap",2,226),("put",2,227),("getId",2,227),("update",0,228)]])
# ---- 6 PetRestControllerV1.updatePet
p = RC + "PetRestControllerV1.java"; E = "PetRestControllerV1.java:"
k = K(77, "currentPet == null")
M[6] = (p, p + "::PetRestControllerV1.updatePet", [
 ("data", P(0,"petId"), C("findPetById",0,76), E+"76"),
 ("data", C("findPetById",0,76), k, E+"76-77"),
 ] + [("data", C("findPetById",0,76), C(n,0,l), E+f"currentPet to {n}, {l}") for n,l in
      [("setBirthDate",80),("setName",81),("setType",82),("savePet",83),("toPetDto",84)]] + [
 ("data", P(1,"petDto"), C("getBirthDate",0,80), E+"80"),
 ("data", P(1,"petDto"), C("getName",0,81), E+"81"),
 ("data", P(1,"petDto"), C("getType",0,82), E+"82"),
 ("data", C("getBirthDate",0,80), C("setBirthDate",0,80), E+"80"),
 ("data", C("getName",0,81), C("setName",0,81), E+"81"),
 ("data", C("getType",0,82), C("toPetType",0,82), E+"82"),
 ("data", C("toPetType",0,82), C("setType",0,82), E+"82"),
 ("data", C("ResponseEntity",0,78), R(0,78), E+"78"),
 ("data", C("toPetDto",0,84), C("ResponseEntity",1,84), E+"84"),
 ("data", C("ResponseEntity",1,84), R(1,84), E+"84"),
 ("control", k, C("ResponseEntity",0,78), E+"77-78"),
 ("control", k, R(0,78), E+"77-78"),
 ] + [("control", k, C(n,o,l), E+f"skipped by early return 77-78, {n}@{l}") for n,o,l in
      [("getBirthDate",0,80),("setBirthDate",0,80),("getName",0,81),("setName",0,81),("getType",0,82),
       ("toPetType",0,82),("setType",0,82),("savePet",0,83),("toPetDto",0,84),("ResponseEntity",1,84)]]
   + [("control", k, R(1,84), E+"skipped by early return 77-78")])
# ---- 26 SpecialtyRestControllerV1.deleteSpecialty
p = RC + "SpecialtyRestControllerV1.java"; E = "SpecialtyRestControllerV1.java:"
k = K(101, "specialty == null")
M[26] = (p, p + "::SpecialtyRestControllerV1.deleteSpecialty", [
 ("data", P(0,"specialtyId"), C("findSpecialtyById",0,100), E+"100"),
 ("data", C("findSpecialtyById",0,100), k, E+"100-101"),
 ("data", C("findSpecialtyById",0,100), C("deleteSpecialty",0,104), E+"100,104"),
 ("data", C("ResponseEntity",0,102), R(0,102), E+"102"),
 ("data", C("ResponseEntity",1,105), R(1,105), E+"105"),
 ("control", k, C("ResponseEntity",0,102), E+"101-102"),
 ("control", k, R(0,102), E+"101-102"),
 ("control", k, C("deleteSpecialty",0,104), E+"skipped by early return"),
 ("control", k, C("ResponseEntity",1,105), E+"skipped by early return"),
 ("control", k, R(1,105), E+"skipped by early return")])
# ---- 19 JdbcVetRepositoryImpl.save
p = JD + "JdbcVetRepositoryImpl.java"; E = "JdbcVetRepositoryImpl.java:"
k = K(143, "vet.isNew()")
M[19] = (p, p + "::JdbcVetRepositoryImpl.save", [
 ("data", P(0,"vet"), C("BeanPropertySqlParameterSource",0,142), E+"142"),
 ("data", P(0,"vet"), C("isNew",0,143), E+"143"),
 ("data", P(0,"vet"), C("setId",0,145), E+"145"),
 ("data", P(0,"vet"), C("updateVetSpecialties",0,146), E+"146"),
 ("data", P(0,"vet"), C("updateVetSpecialties",1,150), E+"150"),
 ("data", C("BeanPropertySqlParameterSource",0,142), C("executeAndReturnKey",0,144), E+"142,144"),
 ("data", C("BeanPropertySqlParameterSource",0,142), C("update",0,149), E+"142,148-149"),
 ("data", C("isNew",0,143), k, E+"143"),
 ("data", C("executeAndReturnKey",0,144), C("intValue",0,145), E+"144-145 newKey"),
 ("data", C("intValue",0,145), C("setId",0,145), E+"145"),
 ] + [("control", k, C(n,o,l), E+f"{n}@{l} in if/else 143-151") for n,o,l in
      [("executeAndReturnKey",0,144),("intValue",0,145),("setId",0,145),("updateVetSpecialties",0,146),
       ("update",0,149),("updateVetSpecialties",1,150)]])
# ---- 25 JdbcOwnerRepositoryImpl.findAll
p = JD + "JdbcOwnerRepositoryImpl.java"; E = "JdbcOwnerRepositoryImpl.java:"
k = K(210, "total == null")
M[25] = (p, p + "::JdbcOwnerRepositoryImpl.findAll", [
 ("data", P(0,"pageable"), C("getPageSize",0,198), E+"198"),
 ("data", P(0,"pageable"), C("getOffset",0,199), E+"199"),
 ("data", P(0,"pageable"), C("PageImpl",0,210), E+"210"),
 ("data", C("HashMap",0,197), C("put",0,198), E+"197-198"),
 ("data", C("HashMap",0,197), C("put",1,199), E+"197,199"),
 ("data", C("HashMap",0,197), C("query",0,200), E+"197,202"),
 ("data", C("HashMap",0,197), C("queryForObject",0,205), E+"197,207"),
 ("data", C("getPageSize",0,198), C("put",0,198), E+"198"),
 ("data", C("getOffset",0,199), C("put",1,199), E+"199"),
 ("data", C("newInstance",0,203), C("query",0,200), E+"203 arg"),
 ("data", C("query",0,200), C("loadOwnersPetsAndVisits",0,204), E+"200,204 owners"),
 ("data", C("query",0,200), C("PageImpl",0,210), E+"200,210 owners"),
 ("data", C("queryForObject",0,205), k, E+"205,210"),
 ("data", C("queryForObject",0,205), C("PageImpl",0,210), E+"205,210 total in ternary"),
 ("data", C("PageImpl",0,210), R(0,210), E+"210")])
# ---- 28 UserServiceImpl.saveUser
p = "src/main/java/org/springframework/samples/petclinic/service/UserServiceImpl.java"; E = "UserServiceImpl.java:"
ka = K(20, "user.getRoles() == null"); kh = K(20, "user.getRoles() == null || user.getRoles().isEmpty()")
kl = K(24, "Role role : user.getRoles()"); k25 = K(25, '!role.getName().startsWith("ROLE_")'); k29 = K(29, "role.getUser() == null")
M[28] = (p, p + "::UserServiceImpl.saveUser", [
 ("data", P(0,"user"), C("getRoles",0,20), E+"20"),
 ("data", P(0,"user"), C("getRoles",1,20), E+"20"),
 ("data", P(0,"user"), C("getRoles",2,24), E+"24"),
 ("data", P(0,"user"), C("setUser",0,30), E+"30"),
 ("data", P(0,"user"), C("save",0,34), E+"34"),
 ("data", C("getRoles",0,20), ka, E+"20 operand"),
 ("data", C("getRoles",1,20), C("isEmpty",0,20), E+"20"),
 ("data", C("isEmpty",0,20), kh, E+"20"),
 ("data", C("getRoles",2,24), kl, E+"24"),
 ("data", C("getRoles",2,24), C("getName",0,25), E+"role loop var 24-25"),
 ("data", C("getRoles",2,24), C("setName",0,26), E+"role loop var 24,26"),
 ("data", C("getRoles",2,24), C("getName",1,26), E+"role loop var 24,26"),
 ("data", C("getRoles",2,24), C("getUser",0,29), E+"role loop var 24,29"),
 ("data", C("getRoles",2,24), C("setUser",0,30), E+"role loop var 24,30"),
 ("data", C("getName",0,25), C("startsWith",0,25), E+"25"),
 ("data", C("startsWith",0,25), k25, E+"25"),
 ("data", C("getName",1,26), C("setName",0,26), E+"26 concat"),
 ("data", C("getUser",0,29), k29, E+"29"),
 ("control", ka, C("isEmpty",0,20), E+"20 short-circuit"),
 ("control", kh, C("IllegalArgumentException",0,21), E+"20-21"),
 ("control", kh, C("getRoles",2,24), E+"throw at 21 skips 24-34"),
 ("control", kh, kl, E+"throw at 21 skips loop"),
 ("control", kh, C("save",0,34), E+"throw at 21 skips 34"),
 ("control", kl, k25, E+"loop body 24-32"),
 ("control", kl, k29, E+"loop body 24-32"),
 ("control", kl, C("getName",0,25), E+"loop body"),
 ("control", kl, C("startsWith",0,25), E+"loop body"),
 ("control", kl, C("getUser",0,29), E+"loop body"),
 ("control", k25, C("setName",0,26), E+"25-26"),
 ("control", k25, C("getName",1,26), E+"25-26"),
 ("control", k29, C("setUser",0,30), E+"29-30")])
with open(f"{FF}/adjudication/java-flow-relabel.jsonl", "w") as f:
    for o in sorted(M):
        path, qn, pairs = M[o]
        d = build(o, path, pairs); d["qname"] = qn
        f.write(json.dumps(d) + "\n")
print({o: len(M[o][2]) for o in sorted(M)})
