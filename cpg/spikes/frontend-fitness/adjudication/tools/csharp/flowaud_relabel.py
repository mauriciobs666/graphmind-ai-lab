#!/usr/bin/env python3
"""Blind relabel (from sample + flow brief only) of the 6 F2 methods; hand-written pairs."""
import json

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
OUT=(_CPG+"/spikes/frontend-fitness/adjudication/csharp-flow-relabel.jsonl")
P=lambda i,n:{"point":"param","index":i,"name":n}
C=lambda n,o,l:{"point":"call","calleeName":n,"ordinal":o,"line":l}
R=lambda o,l:{"point":"ret","ordinal":o,"line":l}
K=lambda l,c:{"point":"cond","line":l,"code":c}
def d(s,t,e,c="sure"):return {"kind":"data","source":s,"sink":t,"confidence":c,"evidence":e}
def k(s,t,e,c="sure"):return {"kind":"control","source":s,"sink":t,"confidence":c,"evidence":e}
M=[]
# 11
f="MultiplexingMiddleware.cs"
c1=K(246,"route.DownstreamRoute.Count == 1"); c2=K(250,"route.DownstreamRouteConfig?.Count > 0")
c3=K(252,"i < contexts.Count && i < route.DownstreamRouteConfig.Count")
g=C("Get",0,259); a=C("Aggregate",0,260)
M.append({"ordinal":11,"path":"src/Multiplexer/MultiplexingMiddleware.cs","qname":"src/Multiplexer/MultiplexingMiddleware.cs::MultiplexingMiddleware.MapAsync","pairs":[
 d(P(1,"route"),c1,f+":246 route read in condition"),
 d(P(1,"route"),c2,f+":250 route read in condition"),
 d(P(1,"route"),c3,f+":252 route read in for header"),
 d(P(2,"contexts"),c3,f+":252 contexts.Count in for header"),
 d(P(1,"route"),g,f+":259 route passed to Get"),
 d(P(1,"route"),a,f+":260 route passed to Aggregate"),
 d(P(0,"context"),a,f+":260 context passed to Aggregate"),
 d(P(2,"contexts"),a,f+":260 contexts passed to Aggregate"),
 d(g,a,f+":259-260 aggregator = Get(route); aggregator.Aggregate receiver"),
 d(a,R(1,260),f+":260 return aggregator.Aggregate(...)"),
 k(c1,R(0,247),f+":246-247 if Count==1 return"),
 k(c1,c2,f+":246-250 early return skips later condition"),
 k(c1,g,f+":246-259 early return skips Get"),
 k(c1,a,f+":246-260 early return skips Aggregate"),
 k(c1,R(1,260),f+":246-260 early return skips later return"),
 k(c2,c3,f+":250-252 for loop inside if block"),
 ],"unlisted":[{"line":250,"construct":"optional chaining ?."}]})
# 15
f="ServiceEventListener.cs"
cs=K(78,"EventLevel.Informational"); ts=C("ToString",0,79)
M.append({"ordinal":15,"path":"samples/ServiceFabric/ApiGateway/ServiceEventListener.cs","qname":"samples/ServiceFabric/ApiGateway/ServiceEventListener.cs::ServiceEventListener.ConvertLevelToString","pairs":[
 d(P(0,"level"),ts,f+":79 level.ToString() receiver"),
 d(ts,R(0,76),f+":76-79 switch expression result returned"),
 d(P(0,"level"),cs,f+":76 level is the switch governed value tested by case label :78","unsure"),
 k(cs,ts,f+":78-79 arm _ taken only when Informational arm did not match","unsure"),
 ],"unlisted":[{"line":79,"construct":"switch-expression discard arm _ (default, no test) not treated as cond"}]})
# 18
f="ClaimsParser.cs"
cv=K(48,"claimValues.Length > 0")
W=C("Where",0,46);S=C("Select",0,46);T=C("ToArray",0,46)
sv=C("StringValues",0,50);ts=C("ToString",0,50);ok=C("OkResponse",0,50)
ce=C("CannotFindClaimError",0,53);er=C("ErrorResponse",0,53)
M.append({"ordinal":18,"path":"src/Infrastructure/Claims/ClaimsParser.cs","qname":"src/Infrastructure/Claims/ClaimsParser.cs::ClaimsParser.GetValue","pairs":[
 d(P(0,"claims"),W,f+":46 claims.Where receiver"),
 d(W,S,f+":46 chained receiver"),
 d(S,T,f+":46 chained receiver"),
 d(T,cv,f+":46,48 claimValues.Length tested"),
 d(T,sv,f+":46,50 claimValues passed to new StringValues"),
 d(sv,ts,f+":50 StringValues(...).ToString() receiver"),
 d(ts,ok,f+":50 argument to new OkResponse"),
 d(ok,R(0,50),f+":50 return new OkResponse"),
 d(P(1,"key"),ce,f+":53 key interpolated into argument of CannotFindClaimError"),
 d(ce,er,f+":53 argument to new ErrorResponse"),
 d(er,R(1,53),f+":53 return new ErrorResponse"),
 k(cv,sv,f+":48-50 inside if body"),
 k(cv,ts,f+":48-50 inside if body"),
 k(cv,ok,f+":48-50 inside if body"),
 k(cv,R(0,50),f+":48-50 inside if body"),
 k(cv,ce,f+":48-53 if body returns, skips later point"),
 k(cv,er,f+":48-53 if body returns, skips later point"),
 k(cv,R(1,53),f+":48-53 if body returns, skips later return"),
 ],"unlisted":[]})
# 19
f="OcelotLoggerFactory.cs"
c33=K(33,"_disposed");c36=K(36,"disposing")
M.append({"ordinal":19,"path":"src/Logging/OcelotLoggerFactory.cs","qname":"src/Logging/OcelotLoggerFactory.cs::OcelotLoggerFactory.Dispose","pairs":[
 d(P(0,"disposing"),c36,f+":36 disposing tested"),
 k(c33,c36,f+":33-36 early return skips later condition"),
 k(c36,C("Dispose",0,38),f+":36-38 inside if body"),
 ],"unlisted":[]})
# 26
f="AuthenticationMiddleware.cs"
fe=K(66,"scheme in notEmptySchemes");c71=K(71,"result?.Succeeded == true");c83=K(83,"result")
W=C("Where",0,64);AU=C("AuthenticateAsync",0,70);LW=C("LogWarning",0,78);NR=C("NoResult",0,83)
M.append({"ordinal":26,"path":"src/Authentication/AuthenticationMiddleware.cs","qname":"src/Authentication/AuthenticationMiddleware.cs::AuthenticationMiddleware.AuthenticateAsync","pairs":[
 d(P(1,"route"),W,f+":63-64 route...Where receiver chain"),
 d(W,fe,f+":64,66 notEmptySchemes iterated in foreach header"),
 d(W,AU,f+":66,70 loop variable scheme takes Where results, passed to AuthenticateAsync"),
 d(P(0,"context"),AU,f+":70 context.AuthenticateAsync receiver"),
 d(AU,c71,f+":70-71 result tested"),
 d(AU,R(0,73),f+":70,73 return result"),
 d(AU,c83,f+":70,83 result is operand of ??"),
 d(AU,R(1,83),f+":70,83 return result ?? ..."),
 d(NR,R(1,83),f+":83 right operand of ?? returned"),
 k(fe,AU,f+":66-70 loop body"),
 k(fe,c71,f+":66-71 loop body"),
 k(fe,LW,f+":66,76-78 catch inside loop body"),
 k(c71,R(0,73),f+":71-73 inside if body"),
 k(c83,NR,f+":83 result ?? NoResult(): result guards NoResult"),
 ],"unlisted":[{"line":71,"construct":"optional chaining ?."},{"line":76,"construct":"catch without when filter / exception edge"}]})
# 28
f="DiskFileConfigurationRepository.cs"
c105=K(105,"_environmentFile.Exists");c110=K(110,"_ocelotFile.Exists")
SO=C("SerializeObject",0,101);WA=C("WaitAsync",0,102)
W0=C("WriteAllTextAsync",0,108);W1=C("WriteAllTextAsync",1,113)
M.append({"ordinal":28,"path":"src/Configuration/Repository/DiskFileConfigurationRepository.cs","qname":"src/Configuration/Repository/DiskFileConfigurationRepository.cs::DiskFileConfigurationRepository.SetAsync","pairs":[
 d(P(0,"configuration"),SO,f+":101 configuration serialized"),
 d(SO,W0,f+":101,108 json written"),
 d(SO,W1,f+":101,113 json written"),
 d(P(1,"cancellationToken"),WA,f+":102 passed to WaitAsync"),
 d(P(1,"cancellationToken"),W0,f+":108 passed to WriteAllTextAsync"),
 d(P(1,"cancellationToken"),W1,f+":113 passed to WriteAllTextAsync"),
 k(c105,C("Delete",0,106),f+":105-106 if body"),
 k(c110,C("Delete",1,111),f+":110-111 if body"),
 ],"unlisted":[{"line":115,"construct":"finally"}]})
with open(OUT,"w") as o:
    for m in sorted(M,key=lambda m:m["ordinal"]): o.write(json.dumps(m)+"\n")
print([ (m["ordinal"],len(m["pairs"])) for m in M])
