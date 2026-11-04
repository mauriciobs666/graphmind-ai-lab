import re,os,json
R='/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/java/src'
types={}
meths=[]
KW=r'^(if|for|while|switch|return|new|else|catch|throw|try|assert|synchronized)\b'
def nparams(s):
  s=s.strip()
  if not s: return 0
  d=0;n=1
  for c in s:
    if c in '<([': d+=1
    elif c in '>)]': d-=1
    elif c==',' and d==0: n+=1
  return n
for d,_,fs in os.walk(R):
  for f in sorted(fs):
    if not f.endswith('.java'):continue
    p=os.path.join(d,f); rel=os.path.relpath(p,R)
    L=open(p).read().split('\n')
    stack=[];depth=0
    for i,l in enumerate(L,1):
      s=l.strip()
      if s.startswith(('*','//','/*','@','import','package')):
        continue
      m=re.match(r'^(?:(?:public|private|protected|static|final|abstract|sealed)\s+)*(class|interface|enum|record)\s+(\w+)(.*)',s)
      if m:
        stack.append([m.group(2),depth])
        types[m.group(2)]=dict(rel=rel,line=i,qual='.'.join(x[0] for x in stack))
      else:
        m=re.match(r'^(?:(?:public|private|protected|static|final|abstract|default|synchronized|native)\s+)*(?:<[^>]+>\s+)?(?:[\w<>\[\],.?]+\s+)?(\w+)\s*\((.*)',s)
        if m and not re.match(KW,s) and stack:
          name=m.group(1)
          txt=m.group(2); j=i
          while txt.count('(')+1>txt.count(')') and j<len(L): j+=1; txt+=' '+L[j-1].strip()
          dd=1;k=0
          while k<len(txt) and dd>0:
            if txt[k]=='(':dd+=1
            elif txt[k]==')':dd-=1
            k+=1
          params=txt[:k-1]; after=txt[k:]
          j2=j
          while '{' not in after and ';' not in after and j2<len(L): j2+=1; after+=' '+L[j2-1].strip()
          if name==stack[-1][0] or re.match(r'^(?:(?:public|private|protected|static|final|abstract|default|synchronized|native)\s+)*(?:<[^>]+>\s+)?[\w<>\[\],.?]+\s+\w+\s*\(',s):
            meths.append(dict(rel=rel,line=i,type=stack[-1][0],qual='.'.join(x[0] for x in stack),name=name,n=nparams(re.sub(r'@\w+(\([^)]*\))?','',params))))
      depth+=l.count('{')-l.count('}')
      while stack and depth<=stack[-1][1] and '}' in l: stack.pop()
