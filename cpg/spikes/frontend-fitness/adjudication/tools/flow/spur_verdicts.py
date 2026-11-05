# rank -> (verdict, reason)
SP, OM, DUP = "spurious", "omission", "duplicate-of-golden"
V = {}
V["python"] = {
 0:(SP,"operand-not-a-cond"),1:(DUP,"lowered-loop-header"),2:(SP,"synthetic-cond:try"),3:(DUP,"cond-code-text"),4:(SP,"synthetic-cond:try"),
 5:(SP,"synthetic-cond:try"),6:(SP,"synthetic-cond:with-finally"),7:(SP,"loop-carried"),8:(SP,"operand-not-a-cond"),9:(DUP,"lowered-loop-header"),
 10:(SP,"synthetic-cond:genexp"),11:(SP,"not-nearest:header-to-after-loop"),12:(SP,"operand-not-a-cond"),13:(DUP,"cond-code-text"),14:(SP,"operand-not-a-cond"),
 15:(SP,"out-of-method-endpoint"),16:(SP,"short-circuit-operand-to-ret"),17:(SP,"operand-not-a-cond"),18:(DUP,"lowered-loop-header"),19:(SP,"operand-not-a-cond"),
 20:(DUP,"lowered-loop-header"),21:(DUP,"lowered-loop-header"),22:(SP,"not-innermost-cond"),23:(DUP,"lowered-loop-header"),24:(SP,"synthetic-cond:genexp"),
 25:(DUP,"lowered-loop-header"),26:(DUP,"lowered-loop-header"),27:(DUP,"lowered-loop-header"),28:(SP,"operand-not-a-cond"),29:(SP,"not-innermost-cond"),
 30:(SP,"operand-not-a-cond"),31:(DUP,"lowered-loop-header"),32:(SP,"short-circuit-operand-to-ret"),33:(SP,"synthetic-cond:try"),34:(OM,"early-exit-later-point"),
 35:(SP,"synthetic-cond:try"),36:(SP,"loop-carried"),37:(DUP,"lowered-loop-header"),38:(SP,"short-circuit-operand-to-ret"),39:(DUP,"lowered-loop-header"),
 40:(DUP,"lowered-loop-header"),41:(SP,"operand-to-enclosing-cond"),42:(SP,"not-innermost-cond"),43:(SP,"operand-to-enclosing-cond"),44:(SP,"synthetic-cond:try"),
 45:(SP,"synthetic-cond:try"),46:(SP,"operand-to-enclosing-cond"),47:(DUP,"lowered-loop-header"),48:(DUP,"lowered-loop-header"),49:(SP,"not-innermost-cond")}
sw=(DUP,"switch-model:discriminant-vs-case-label")
V["javascript"] = {i:sw for i in (0,2,3,4,5,6,8,9,11,12,13,16,17,18,23,25,26,27,28,29,32,35,37,38,40,41,42,43,48)}
V["javascript"].update({1:(DUP,"cond-code-text"),36:(DUP,"lowered-loop-header"),45:(DUP,"lowered-loop-header"),15:(OM,"cond-decides-ret-in-its-body"),
 7:(SP,"synthetic-cond:null-literal"),19:(SP,"synthetic-cond:null-literal"),24:(SP,"synthetic-cond:null-literal"),39:(SP,"synthetic-cond:null-literal"),
 10:(SP,"synthetic-cond:try"),22:(SP,"synthetic-cond:try"),33:(SP,"synthetic-cond:try"),46:(SP,"synthetic-cond:try"),
 14:(SP,"operand-not-a-cond"),20:(SP,"ternary-selector"),21:(SP,"operand-not-a-cond"),30:(SP,"synthetic-cond:default-param"),31:(SP,"ternary-selector"),
 34:(SP,"ternary-selector"),44:(SP,"ternary-selector"),47:(SP,"operand-to-enclosing-cond")})
V["go"] = {i:(SP,"operand-not-a-cond") for i in (2,3,6,7,8,9,11,12,13,14,16,18,22,24,29,30,31,32,33,37,38,40)}
V["go"].update({1:(SP,"operand-to-enclosing-cond"),36:(SP,"operand-to-enclosing-cond"),4:(SP,"operand-to-enclosing-cond"),5:(SP,"operand-to-enclosing-cond"),
 34:(SP,"operand-to-enclosing-cond"),39:(SP,"operand-to-enclosing-cond"),10:(SP,"operand-not-a-cond"),20:(SP,"loop-carried"),28:(SP,"loop-carried"),
 0:(SP,"not-nearest:enclosing-if-of-nested-early-exit"),35:(SP,"not-nearest:enclosing-if-of-nested-early-exit"),
 15:(OM,"early-exit-later-point"),26:(OM,"early-exit-later-point"),
 17:(SP,"operator-not-call:go-conversion"),19:(SP,"operator-not-call:go-conversion"),25:(SP,"operator-not-call:go-conversion"),
 21:(SP,"operator-not-call:go-conversion"),23:(SP,"operator-not-call:go-conversion"),27:(SP,"operator-not-call:go-conversion")})
V["java"] = {0:(SP,"ternary-selector"),1:(SP,"ternary-selector"),2:(SP,"operand-to-enclosing-cond"),3:(SP,"loop-carried"),4:(SP,"loop-carried"),
 5:(SP,"not-innermost-cond"),6:(SP,"not-innermost-cond"),7:(SP,"ternary-selector"),8:(SP,"ternary-selector"),9:(SP,"ternary-selector"),10:(SP,"ternary-selector"),
 11:(SP,"ternary-selector"),12:(SP,"not-nearest:header-to-after-loop"),13:(SP,"operand-not-a-cond"),14:(SP,"ternary-selector"),15:(SP,"operand-to-enclosing-cond")}
V["csharp"] = {0:(SP,"not-innermost-cond"),1:(SP,"ternary-selector"),2:(SP,"operand-not-a-cond"),3:(SP,"operand-to-enclosing-cond"),4:(SP,"operand-not-a-cond"),
 5:(OM,"short-circuit-nested-call"),6:(SP,"operand-not-a-cond"),7:(SP,"handler-not-decided"),8:(SP,"header-own-iterable"),9:(SP,"ternary-selector"),
 10:(SP,"operand-not-a-cond"),11:(DUP,"lowered-loop-header"),12:(DUP,"lowered-loop-header"),13:(SP,"operator-not-call:csharp-nameof"),14:(SP,"operand-to-enclosing-cond"),
 15:(SP,"operand-not-a-cond"),16:(SP,"synthetic-cond:finally"),17:(SP,"operator-not-call:csharp-nameof"),18:(OM,"early-exit-later-point"),19:(SP,"loop-carried"),
 20:(SP,"ternary-selector"),21:(SP,"not-innermost-cond"),22:(SP,"not-innermost-cond"),23:(SP,"short-circuit-operand-to-ret"),24:(SP,"synthetic-cond:lock"),
 25:(SP,"not-innermost-cond"),26:(DUP,"lowered-loop-header"),27:(SP,"not-innermost-cond"),28:(SP,"operand-not-a-cond"),29:(SP,"not-innermost-cond"),
 30:(DUP,"lowered-loop-header"),31:(DUP,"lowered-loop-header"),32:(SP,"operand-to-enclosing-cond"),33:(OM,"early-exit-later-point"),34:(SP,"synthetic-cond:lock"),
 35:(SP,"ternary-selector"),36:(SP,"operand-to-enclosing-cond"),37:(SP,"operand-not-a-cond"),38:(SP,"short-circuit-operand-to-ret"),39:(SP,"operand-not-a-cond"),
 40:(OM,"short-circuit-nested-call"),41:(DUP,"lowered-loop-header"),42:(OM,"early-exit-later-point"),43:(SP,"operand-to-enclosing-cond"),44:(SP,"ternary-selector"),
 45:(OM,"early-exit-later-point"),46:(SP,"not-nearest:header-to-after-loop"),47:(DUP,"lowered-loop-header"),48:(SP,"operand-to-enclosing-cond"),49:(SP,"synthetic-cond:lock")}
mc=(SP,"macro-internal-cond")
V["cpp"] = {39:(SP,"partF-ruled-invalid:index-as-selector"),34:(SP,"not-innermost-cond"),10:mc,16:mc,26:mc,40:(SP,"loop-carried"),
 1:(SP,"operand-not-a-cond"),17:(SP,"operand-not-a-cond"),35:(SP,"operand-not-a-cond"),11:mc,30:mc,
 7:sw,14:sw,20:sw,24:mc,27:(SP,"ternary-selector"),36:(SP,"ternary-selector"),
 4:(SP,"not-nearest:header-to-after-loop"),6:(SP,"loop-carried"),9:(SP,"loop-carried"),13:(SP,"loop-carried"),19:(SP,"loop-carried"),23:(SP,"loop-carried"),31:(SP,"loop-carried")}
for i in (2,3,5,8,12,15,18,21,25,28,29,32,33,37,38,42,43): V["cpp"][i]=mc
for i in (0,22,44): V["cpp"][i]=(SP,"operand-not-a-cond")
V["cpp"][41]=(SP,"out-of-method-endpoint")
