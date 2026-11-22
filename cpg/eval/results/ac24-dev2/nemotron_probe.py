import sys, json
sys.path[:0]=["/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad/clone/cpg/eval/tools","/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad/clone/cpg"]
import ac24_smoke as A
raw=[]
_orig=A._reply_text
def spy(resp):
    try: raw.append(json.loads(resp.body)["choices"][0])
    except Exception: raw.append(None)
    return _orig(resp)
A._reply_text=spy
rig=A.build_live_rig("graphmind-ai-lab")
from clients.common import ClientError, NotBuilt
bad=0
for i,q in enumerate(A.CANARY_QUESTIONS):
    m=rig.log.mark()
    try: rig.client.ask(question=q, cache_bypass=True)
    except NotBuilt: pass
    except ClientError as e: print(i,"clienterror",e.code)
    ev=rig.log.events_since(m)
    ch=raw[-1] if raw else None
    msg=(ch or {}).get("message",{})
    content=msg.get("content") or ""
    think = bool(msg.get("reasoning_content")) or "<think" in content.lower() or "</think" in content.lower() or (content.strip()[:1] not in ("{",))
    try: json.loads(content); isjson=True
    except Exception: isjson=False
    print(i,"finish_reason=",(ch or {}).get("finish_reason"),"json=",isjson,"thinking_text=",think,"reasoning_content_len=",len(msg.get("reasoning_content") or ""),"content_len=",len(content),"draftNone=",[e["draftNone"] for e in ev])
    if (ch or {}).get("finish_reason")=="length" or think or not isjson or any(e["draftNone"] for e in ev): bad+=1
print("PROBE", "FAIL" if bad else "PASS", "bad=",bad)
