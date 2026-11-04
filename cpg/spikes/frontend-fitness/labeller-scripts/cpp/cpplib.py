import json,os
S='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/sample/cpp.jsonl'
O='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/labels/cpp.jsonl'
sites={}
for l in open(S):
    d=json.loads(l); sites[d['ordinal']]=d
def I(p,q,l): return {"kind":"internal","path":p,"qname":p+"::"+q if q else None,"defLine":l}
def D(p,q,l): return {"kind":"declared","path":p,"qname":p+"::"+q if q else None,"defLine":l}
def X(pk): return {"kind":"external","package":pk}
GT='googletest/src/gtest.cc'
AR='googletest/include/gtest/gtest-assertion-result.h'
GH='googletest/include/gtest/gtest.h'
GM='googlemock/include/gmock/gmock-matchers.h'
GMT='googletest/include/gtest/gtest-matchers.h'
CARD='googlemock/include/gmock/gmock-cardinalities.h'
SB='googlemock/include/gmock/gmock-spec-builders.h'
GA='googlemock/include/gmock/gmock-actions.h'
AH_CTOR=I(GT,'testing.internal.AssertHelper.<init>',499)
AH_ASSIGN=I(GT,'testing.internal.AssertHelper.operator=',515)
MSG=I(GT,'testing.Message.<init>',1345)
AR_FM=I(AR,'testing.AssertionResult.failure_message',192)
CORE=[AH_CTOR,AH_ASSIGN,MSG,AR_FM]
BOOL=[I(GT,'testing.internal.GetBoolAssertionFailureMessage',1695),AH_CTOR,AH_ASSIGN,MSG]
THAT=[I(GM,'testing.internal.MakePredicateFormatterFromMatcher',1737)]+CORE
EQ=[I(GH,'testing.internal.EqHelper.Compare',1415),I(GH,'testing.internal.EqHelper.Compare#2',1427),I(GH,'testing.internal.EqHelper.Compare#3',1434)]+CORE
EQE='EXPECT_EQ gtest.h:1898 -> EXPECT_PRED_FORMAT2 -> GTEST_PRED_FORMAT2_/GTEST_ASSERT_ (gtest_pred_impl.h:77,133); Compare overload not determined; AssertionResult ctor omitted'
BOOLE='EXPECT_TRUE/FALSE gtest.h:1837 -> GTEST_TEST_BOOLEAN_ (gtest-internal.h:1490) -> GetBoolAssertionFailureMessage, AssertHelper, Message; AssertionResult ctor omitted'
THATE='EXPECT_THAT gmock-matchers.h:5906 -> EXPECT_PRED_FORMAT1 + MakePredicateFormatterFromMatcher:1737; GTEST_ASSERT_'
def last():
    n=0
    if os.path.exists(O):
        for l in open(O): n=json.loads(l)['ordinal']
    return n
def count():
    c=0
    if os.path.exists(O):
        for l in open(O):
            d=json.loads(l)
            if any(x['kind'] in('internal','declared') for x in d['callees']): c+=1
    return c
def W(o,callees,conf='sure',ev='',notACall=False,dynamic=False):
    assert o==last()+1,(o,last())
    r={"ordinal":o,"siteId":sites[o]['siteId'],"callees":callees,"notACall":notACall,"dynamic":dynamic,"confidence":conf,"evidence":ev[:300]}
    open(O,'a').write(json.dumps(r)+'\n')
SPI=[I(GT,'testing.internal.SingleFailureChecker.<init>',1080),I(GT,'testing.ScopedFakeTestPartResultReporter.<init>',986),I(GT,'testing.internal.AlwaysTrue',6528)]
SPIE='EXPECT_(NON)FATAL_FAILURE gtest-spi.h:149/217 -> SingleFailureChecker ctor gtest.cc:1080, ScopedFakeTestPartResultReporter ctor :986, AlwaysTrue; nested statement calls are other sites'
FAILE='ADD_FAILURE/FAIL gtest.h:1765/1774 -> GTEST_(NON)FATAL_FAILURE_ -> AssertHelper ctor/operator=, Message ctor (gtest-internal.h:1328)'
FAIL3=[AH_CTOR,AH_ASSIGN,MSG]
GTL=I('googletest/src/gtest-port.cc','testing.internal.GTestLog.<init>',845)
DEATH=[I('googletest/src/gtest-death-test.cc','testing.internal.DeathTest.Create',388),I('googletest/include/gtest/internal/gtest-death-test-internal.h','testing.internal.MakeDeathTestMatcher',63),I(GT,'testing.internal.AlwaysTrue',6528)]+CORE
FPI='googletest/include/gtest/internal/gtest-filepath.h'
FPC=I(FPI,'testing.internal.FilePath.<init>',74)
