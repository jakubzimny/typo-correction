��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.12v2.3.0-54-gfcc4b966f18�
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:72*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:72*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:2*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:2(*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:(*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(#*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:(#*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:#*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	#�*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	#�*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:�*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�#*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	�#*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:#*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#(*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:#(*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:(*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(-*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:(-*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:-*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-7*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:-7*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:7*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:72*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:72*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:2*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:2(*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:(*
dtype0
�
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(#*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:(#*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:#*
dtype0
�
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	#�*&
shared_nameAdam/dense_3/kernel/m
�
)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes
:	#�*
dtype0

Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/dense_3/bias/m
x
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�#*&
shared_nameAdam/dense_4/kernel/m
�
)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes
:	�#*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:#*
dtype0
�
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#(*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:#(*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:(*
dtype0
�
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(-*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

:(-*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:-*
dtype0
�
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-7*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

:-7*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:7*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:72*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:72*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:2*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:2(*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:(*
dtype0
�
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(#*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:(#*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:#*
dtype0
�
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	#�*&
shared_nameAdam/dense_3/kernel/v
�
)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes
:	#�*
dtype0

Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/dense_3/bias/v
x
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�#*&
shared_nameAdam/dense_4/kernel/v
�
)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes
:	�#*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:#*
dtype0
�
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#(*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:#(*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:(*
dtype0
�
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(-*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

:(-*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:-*
dtype0
�
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-7*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

:-7*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:7*
dtype0

NoOpNoOp
�Q
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�P
value�PB�P B�P
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
h

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
�
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_ratem}m~mm�m�m�"m�#m�(m�)m�.m�/m�4m�5m�:m�;m�v�v�v�v�v�v�"v�#v�(v�)v�.v�/v�4v�5v�:v�;v�
 
v
0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15
v
0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15
�
Emetrics
regularization_losses
	variables
Fnon_trainable_variables

Glayers
trainable_variables
Hlayer_metrics
Ilayer_regularization_losses
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
Jmetrics
regularization_losses
	variables
Knon_trainable_variables

Llayers
trainable_variables
Mlayer_metrics
Nlayer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
Ometrics
regularization_losses
	variables
Pnon_trainable_variables

Qlayers
trainable_variables
Rlayer_metrics
Slayer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
Tmetrics
regularization_losses
	variables
Unon_trainable_variables

Vlayers
 trainable_variables
Wlayer_metrics
Xlayer_regularization_losses
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
�
Ymetrics
$regularization_losses
%	variables
Znon_trainable_variables

[layers
&trainable_variables
\layer_metrics
]layer_regularization_losses
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
�
^metrics
*regularization_losses
+	variables
_non_trainable_variables

`layers
,trainable_variables
alayer_metrics
blayer_regularization_losses
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
�
cmetrics
0regularization_losses
1	variables
dnon_trainable_variables

elayers
2trainable_variables
flayer_metrics
glayer_regularization_losses
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
�
hmetrics
6regularization_losses
7	variables
inon_trainable_variables

jlayers
8trainable_variables
klayer_metrics
llayer_regularization_losses
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

:0
;1
�
mmetrics
<regularization_losses
=	variables
nnon_trainable_variables

olayers
>trainable_variables
player_metrics
qlayer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

r0
s1
 
?
0
1
2
3
4
5
6
7
	8
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ttotal
	ucount
v	variables
w	keras_api
D
	xtotal
	ycount
z
_fn_kwargs
{	variables
|	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

t0
u1

v	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

x0
y1

{	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*+
_output_shapes
:���������7*
dtype0* 
shape:���������7
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������7*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_89485
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_90513
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_90694��
�
�
B__inference_dense_6_layer_call_and_return_conditional_losses_89167

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:(-*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������(2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:-2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������-2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������-2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������-2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������-2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������(:::S O
+
_output_shapes
:���������(
 
_user_specified_nameinputs
�
|
'__inference_dense_3_layer_call_fn_90159

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_890262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������#::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������#
 
_user_specified_nameinputs
�
�
B__inference_dense_5_layer_call_and_return_conditional_losses_89120

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:#(*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������#2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������(2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������(2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������#:::S O
+
_output_shapes
:���������#
 
_user_specified_nameinputs
�
�
B__inference_dense_4_layer_call_and_return_conditional_losses_90190

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�#*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:����������2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:#*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������#2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������#2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������:::T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�)
�
G__inference_functional_1_layer_call_and_return_conditional_losses_89322

inputs
dense_89281
dense_89283
dense_1_89286
dense_1_89288
dense_2_89291
dense_2_89293
dense_3_89296
dense_3_89298
dense_4_89301
dense_4_89303
dense_5_89306
dense_5_89308
dense_6_89311
dense_6_89313
dense_7_89316
dense_7_89318
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_89281dense_89283*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_888852
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_89286dense_1_89288*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_889322!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_89291dense_2_89293*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_889792!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_89296dense_3_89298*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_890262!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_89301dense_4_89303*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_890732!
dense_4/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_89306dense_5_89308*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_891202!
dense_5/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_89311dense_6_89313*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_891672!
dense_6/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_89316dense_7_89318*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������7*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_892142!
dense_7/StatefulPartitionedCall�
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������7::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:S O
+
_output_shapes
:���������7
 
_user_specified_nameinputs
��
�
 __inference__wrapped_model_88850
input_18
4functional_1_dense_tensordot_readvariableop_resource6
2functional_1_dense_biasadd_readvariableop_resource:
6functional_1_dense_1_tensordot_readvariableop_resource8
4functional_1_dense_1_biasadd_readvariableop_resource:
6functional_1_dense_2_tensordot_readvariableop_resource8
4functional_1_dense_2_biasadd_readvariableop_resource:
6functional_1_dense_3_tensordot_readvariableop_resource8
4functional_1_dense_3_biasadd_readvariableop_resource:
6functional_1_dense_4_tensordot_readvariableop_resource8
4functional_1_dense_4_biasadd_readvariableop_resource:
6functional_1_dense_5_tensordot_readvariableop_resource8
4functional_1_dense_5_biasadd_readvariableop_resource:
6functional_1_dense_6_tensordot_readvariableop_resource8
4functional_1_dense_6_biasadd_readvariableop_resource:
6functional_1_dense_7_tensordot_readvariableop_resource8
4functional_1_dense_7_biasadd_readvariableop_resource
identity��
+functional_1/dense/Tensordot/ReadVariableOpReadVariableOp4functional_1_dense_tensordot_readvariableop_resource*
_output_shapes

:72*
dtype02-
+functional_1/dense/Tensordot/ReadVariableOp�
!functional_1/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!functional_1/dense/Tensordot/axes�
!functional_1/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2#
!functional_1/dense/Tensordot/free
"functional_1/dense/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
:2$
"functional_1/dense/Tensordot/Shape�
*functional_1/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_1/dense/Tensordot/GatherV2/axis�
%functional_1/dense/Tensordot/GatherV2GatherV2+functional_1/dense/Tensordot/Shape:output:0*functional_1/dense/Tensordot/free:output:03functional_1/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%functional_1/dense/Tensordot/GatherV2�
,functional_1/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense/Tensordot/GatherV2_1/axis�
'functional_1/dense/Tensordot/GatherV2_1GatherV2+functional_1/dense/Tensordot/Shape:output:0*functional_1/dense/Tensordot/axes:output:05functional_1/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'functional_1/dense/Tensordot/GatherV2_1�
"functional_1/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"functional_1/dense/Tensordot/Const�
!functional_1/dense/Tensordot/ProdProd.functional_1/dense/Tensordot/GatherV2:output:0+functional_1/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!functional_1/dense/Tensordot/Prod�
$functional_1/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$functional_1/dense/Tensordot/Const_1�
#functional_1/dense/Tensordot/Prod_1Prod0functional_1/dense/Tensordot/GatherV2_1:output:0-functional_1/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#functional_1/dense/Tensordot/Prod_1�
(functional_1/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(functional_1/dense/Tensordot/concat/axis�
#functional_1/dense/Tensordot/concatConcatV2*functional_1/dense/Tensordot/free:output:0*functional_1/dense/Tensordot/axes:output:01functional_1/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#functional_1/dense/Tensordot/concat�
"functional_1/dense/Tensordot/stackPack*functional_1/dense/Tensordot/Prod:output:0,functional_1/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"functional_1/dense/Tensordot/stack�
&functional_1/dense/Tensordot/transpose	Transposeinput_1,functional_1/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������72(
&functional_1/dense/Tensordot/transpose�
$functional_1/dense/Tensordot/ReshapeReshape*functional_1/dense/Tensordot/transpose:y:0+functional_1/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2&
$functional_1/dense/Tensordot/Reshape�
#functional_1/dense/Tensordot/MatMulMatMul-functional_1/dense/Tensordot/Reshape:output:03functional_1/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22%
#functional_1/dense/Tensordot/MatMul�
$functional_1/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:22&
$functional_1/dense/Tensordot/Const_2�
*functional_1/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_1/dense/Tensordot/concat_1/axis�
%functional_1/dense/Tensordot/concat_1ConcatV2.functional_1/dense/Tensordot/GatherV2:output:0-functional_1/dense/Tensordot/Const_2:output:03functional_1/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%functional_1/dense/Tensordot/concat_1�
functional_1/dense/TensordotReshape-functional_1/dense/Tensordot/MatMul:product:0.functional_1/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������22
functional_1/dense/Tensordot�
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp2functional_1_dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOp�
functional_1/dense/BiasAddBiasAdd%functional_1/dense/Tensordot:output:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������22
functional_1/dense/BiasAdd�
functional_1/dense/ReluRelu#functional_1/dense/BiasAdd:output:0*
T0*+
_output_shapes
:���������22
functional_1/dense/Relu�
-functional_1/dense_1/Tensordot/ReadVariableOpReadVariableOp6functional_1_dense_1_tensordot_readvariableop_resource*
_output_shapes

:2(*
dtype02/
-functional_1/dense_1/Tensordot/ReadVariableOp�
#functional_1/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#functional_1/dense_1/Tensordot/axes�
#functional_1/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#functional_1/dense_1/Tensordot/free�
$functional_1/dense_1/Tensordot/ShapeShape%functional_1/dense/Relu:activations:0*
T0*
_output_shapes
:2&
$functional_1/dense_1/Tensordot/Shape�
,functional_1/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense_1/Tensordot/GatherV2/axis�
'functional_1/dense_1/Tensordot/GatherV2GatherV2-functional_1/dense_1/Tensordot/Shape:output:0,functional_1/dense_1/Tensordot/free:output:05functional_1/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'functional_1/dense_1/Tensordot/GatherV2�
.functional_1/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.functional_1/dense_1/Tensordot/GatherV2_1/axis�
)functional_1/dense_1/Tensordot/GatherV2_1GatherV2-functional_1/dense_1/Tensordot/Shape:output:0,functional_1/dense_1/Tensordot/axes:output:07functional_1/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)functional_1/dense_1/Tensordot/GatherV2_1�
$functional_1/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$functional_1/dense_1/Tensordot/Const�
#functional_1/dense_1/Tensordot/ProdProd0functional_1/dense_1/Tensordot/GatherV2:output:0-functional_1/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#functional_1/dense_1/Tensordot/Prod�
&functional_1/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&functional_1/dense_1/Tensordot/Const_1�
%functional_1/dense_1/Tensordot/Prod_1Prod2functional_1/dense_1/Tensordot/GatherV2_1:output:0/functional_1/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%functional_1/dense_1/Tensordot/Prod_1�
*functional_1/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_1/dense_1/Tensordot/concat/axis�
%functional_1/dense_1/Tensordot/concatConcatV2,functional_1/dense_1/Tensordot/free:output:0,functional_1/dense_1/Tensordot/axes:output:03functional_1/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%functional_1/dense_1/Tensordot/concat�
$functional_1/dense_1/Tensordot/stackPack,functional_1/dense_1/Tensordot/Prod:output:0.functional_1/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$functional_1/dense_1/Tensordot/stack�
(functional_1/dense_1/Tensordot/transpose	Transpose%functional_1/dense/Relu:activations:0.functional_1/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������22*
(functional_1/dense_1/Tensordot/transpose�
&functional_1/dense_1/Tensordot/ReshapeReshape,functional_1/dense_1/Tensordot/transpose:y:0-functional_1/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2(
&functional_1/dense_1/Tensordot/Reshape�
%functional_1/dense_1/Tensordot/MatMulMatMul/functional_1/dense_1/Tensordot/Reshape:output:05functional_1/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2'
%functional_1/dense_1/Tensordot/MatMul�
&functional_1/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2(
&functional_1/dense_1/Tensordot/Const_2�
,functional_1/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense_1/Tensordot/concat_1/axis�
'functional_1/dense_1/Tensordot/concat_1ConcatV20functional_1/dense_1/Tensordot/GatherV2:output:0/functional_1/dense_1/Tensordot/Const_2:output:05functional_1/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'functional_1/dense_1/Tensordot/concat_1�
functional_1/dense_1/TensordotReshape/functional_1/dense_1/Tensordot/MatMul:product:00functional_1/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������(2 
functional_1/dense_1/Tensordot�
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02-
+functional_1/dense_1/BiasAdd/ReadVariableOp�
functional_1/dense_1/BiasAddBiasAdd'functional_1/dense_1/Tensordot:output:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2
functional_1/dense_1/BiasAdd�
functional_1/dense_1/ReluRelu%functional_1/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������(2
functional_1/dense_1/Relu�
-functional_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6functional_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:(#*
dtype02/
-functional_1/dense_2/Tensordot/ReadVariableOp�
#functional_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#functional_1/dense_2/Tensordot/axes�
#functional_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#functional_1/dense_2/Tensordot/free�
$functional_1/dense_2/Tensordot/ShapeShape'functional_1/dense_1/Relu:activations:0*
T0*
_output_shapes
:2&
$functional_1/dense_2/Tensordot/Shape�
,functional_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense_2/Tensordot/GatherV2/axis�
'functional_1/dense_2/Tensordot/GatherV2GatherV2-functional_1/dense_2/Tensordot/Shape:output:0,functional_1/dense_2/Tensordot/free:output:05functional_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'functional_1/dense_2/Tensordot/GatherV2�
.functional_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.functional_1/dense_2/Tensordot/GatherV2_1/axis�
)functional_1/dense_2/Tensordot/GatherV2_1GatherV2-functional_1/dense_2/Tensordot/Shape:output:0,functional_1/dense_2/Tensordot/axes:output:07functional_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)functional_1/dense_2/Tensordot/GatherV2_1�
$functional_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$functional_1/dense_2/Tensordot/Const�
#functional_1/dense_2/Tensordot/ProdProd0functional_1/dense_2/Tensordot/GatherV2:output:0-functional_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#functional_1/dense_2/Tensordot/Prod�
&functional_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&functional_1/dense_2/Tensordot/Const_1�
%functional_1/dense_2/Tensordot/Prod_1Prod2functional_1/dense_2/Tensordot/GatherV2_1:output:0/functional_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%functional_1/dense_2/Tensordot/Prod_1�
*functional_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_1/dense_2/Tensordot/concat/axis�
%functional_1/dense_2/Tensordot/concatConcatV2,functional_1/dense_2/Tensordot/free:output:0,functional_1/dense_2/Tensordot/axes:output:03functional_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%functional_1/dense_2/Tensordot/concat�
$functional_1/dense_2/Tensordot/stackPack,functional_1/dense_2/Tensordot/Prod:output:0.functional_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$functional_1/dense_2/Tensordot/stack�
(functional_1/dense_2/Tensordot/transpose	Transpose'functional_1/dense_1/Relu:activations:0.functional_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������(2*
(functional_1/dense_2/Tensordot/transpose�
&functional_1/dense_2/Tensordot/ReshapeReshape,functional_1/dense_2/Tensordot/transpose:y:0-functional_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2(
&functional_1/dense_2/Tensordot/Reshape�
%functional_1/dense_2/Tensordot/MatMulMatMul/functional_1/dense_2/Tensordot/Reshape:output:05functional_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#2'
%functional_1/dense_2/Tensordot/MatMul�
&functional_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2(
&functional_1/dense_2/Tensordot/Const_2�
,functional_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense_2/Tensordot/concat_1/axis�
'functional_1/dense_2/Tensordot/concat_1ConcatV20functional_1/dense_2/Tensordot/GatherV2:output:0/functional_1/dense_2/Tensordot/Const_2:output:05functional_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'functional_1/dense_2/Tensordot/concat_1�
functional_1/dense_2/TensordotReshape/functional_1/dense_2/Tensordot/MatMul:product:00functional_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#2 
functional_1/dense_2/Tensordot�
+functional_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype02-
+functional_1/dense_2/BiasAdd/ReadVariableOp�
functional_1/dense_2/BiasAddBiasAdd'functional_1/dense_2/Tensordot:output:03functional_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#2
functional_1/dense_2/BiasAdd�
functional_1/dense_2/ReluRelu%functional_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������#2
functional_1/dense_2/Relu�
-functional_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6functional_1_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	#�*
dtype02/
-functional_1/dense_3/Tensordot/ReadVariableOp�
#functional_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#functional_1/dense_3/Tensordot/axes�
#functional_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#functional_1/dense_3/Tensordot/free�
$functional_1/dense_3/Tensordot/ShapeShape'functional_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2&
$functional_1/dense_3/Tensordot/Shape�
,functional_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense_3/Tensordot/GatherV2/axis�
'functional_1/dense_3/Tensordot/GatherV2GatherV2-functional_1/dense_3/Tensordot/Shape:output:0,functional_1/dense_3/Tensordot/free:output:05functional_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'functional_1/dense_3/Tensordot/GatherV2�
.functional_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.functional_1/dense_3/Tensordot/GatherV2_1/axis�
)functional_1/dense_3/Tensordot/GatherV2_1GatherV2-functional_1/dense_3/Tensordot/Shape:output:0,functional_1/dense_3/Tensordot/axes:output:07functional_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)functional_1/dense_3/Tensordot/GatherV2_1�
$functional_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$functional_1/dense_3/Tensordot/Const�
#functional_1/dense_3/Tensordot/ProdProd0functional_1/dense_3/Tensordot/GatherV2:output:0-functional_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#functional_1/dense_3/Tensordot/Prod�
&functional_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&functional_1/dense_3/Tensordot/Const_1�
%functional_1/dense_3/Tensordot/Prod_1Prod2functional_1/dense_3/Tensordot/GatherV2_1:output:0/functional_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%functional_1/dense_3/Tensordot/Prod_1�
*functional_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_1/dense_3/Tensordot/concat/axis�
%functional_1/dense_3/Tensordot/concatConcatV2,functional_1/dense_3/Tensordot/free:output:0,functional_1/dense_3/Tensordot/axes:output:03functional_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%functional_1/dense_3/Tensordot/concat�
$functional_1/dense_3/Tensordot/stackPack,functional_1/dense_3/Tensordot/Prod:output:0.functional_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$functional_1/dense_3/Tensordot/stack�
(functional_1/dense_3/Tensordot/transpose	Transpose'functional_1/dense_2/Relu:activations:0.functional_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#2*
(functional_1/dense_3/Tensordot/transpose�
&functional_1/dense_3/Tensordot/ReshapeReshape,functional_1/dense_3/Tensordot/transpose:y:0-functional_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2(
&functional_1/dense_3/Tensordot/Reshape�
%functional_1/dense_3/Tensordot/MatMulMatMul/functional_1/dense_3/Tensordot/Reshape:output:05functional_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%functional_1/dense_3/Tensordot/MatMul�
&functional_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2(
&functional_1/dense_3/Tensordot/Const_2�
,functional_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense_3/Tensordot/concat_1/axis�
'functional_1/dense_3/Tensordot/concat_1ConcatV20functional_1/dense_3/Tensordot/GatherV2:output:0/functional_1/dense_3/Tensordot/Const_2:output:05functional_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'functional_1/dense_3/Tensordot/concat_1�
functional_1/dense_3/TensordotReshape/functional_1/dense_3/Tensordot/MatMul:product:00functional_1/dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2 
functional_1/dense_3/Tensordot�
+functional_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+functional_1/dense_3/BiasAdd/ReadVariableOp�
functional_1/dense_3/BiasAddBiasAdd'functional_1/dense_3/Tensordot:output:03functional_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
functional_1/dense_3/BiasAdd�
functional_1/dense_3/ReluRelu%functional_1/dense_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
functional_1/dense_3/Relu�
-functional_1/dense_4/Tensordot/ReadVariableOpReadVariableOp6functional_1_dense_4_tensordot_readvariableop_resource*
_output_shapes
:	�#*
dtype02/
-functional_1/dense_4/Tensordot/ReadVariableOp�
#functional_1/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#functional_1/dense_4/Tensordot/axes�
#functional_1/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#functional_1/dense_4/Tensordot/free�
$functional_1/dense_4/Tensordot/ShapeShape'functional_1/dense_3/Relu:activations:0*
T0*
_output_shapes
:2&
$functional_1/dense_4/Tensordot/Shape�
,functional_1/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense_4/Tensordot/GatherV2/axis�
'functional_1/dense_4/Tensordot/GatherV2GatherV2-functional_1/dense_4/Tensordot/Shape:output:0,functional_1/dense_4/Tensordot/free:output:05functional_1/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'functional_1/dense_4/Tensordot/GatherV2�
.functional_1/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.functional_1/dense_4/Tensordot/GatherV2_1/axis�
)functional_1/dense_4/Tensordot/GatherV2_1GatherV2-functional_1/dense_4/Tensordot/Shape:output:0,functional_1/dense_4/Tensordot/axes:output:07functional_1/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)functional_1/dense_4/Tensordot/GatherV2_1�
$functional_1/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$functional_1/dense_4/Tensordot/Const�
#functional_1/dense_4/Tensordot/ProdProd0functional_1/dense_4/Tensordot/GatherV2:output:0-functional_1/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#functional_1/dense_4/Tensordot/Prod�
&functional_1/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&functional_1/dense_4/Tensordot/Const_1�
%functional_1/dense_4/Tensordot/Prod_1Prod2functional_1/dense_4/Tensordot/GatherV2_1:output:0/functional_1/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%functional_1/dense_4/Tensordot/Prod_1�
*functional_1/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_1/dense_4/Tensordot/concat/axis�
%functional_1/dense_4/Tensordot/concatConcatV2,functional_1/dense_4/Tensordot/free:output:0,functional_1/dense_4/Tensordot/axes:output:03functional_1/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%functional_1/dense_4/Tensordot/concat�
$functional_1/dense_4/Tensordot/stackPack,functional_1/dense_4/Tensordot/Prod:output:0.functional_1/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$functional_1/dense_4/Tensordot/stack�
(functional_1/dense_4/Tensordot/transpose	Transpose'functional_1/dense_3/Relu:activations:0.functional_1/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������2*
(functional_1/dense_4/Tensordot/transpose�
&functional_1/dense_4/Tensordot/ReshapeReshape,functional_1/dense_4/Tensordot/transpose:y:0-functional_1/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2(
&functional_1/dense_4/Tensordot/Reshape�
%functional_1/dense_4/Tensordot/MatMulMatMul/functional_1/dense_4/Tensordot/Reshape:output:05functional_1/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#2'
%functional_1/dense_4/Tensordot/MatMul�
&functional_1/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2(
&functional_1/dense_4/Tensordot/Const_2�
,functional_1/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense_4/Tensordot/concat_1/axis�
'functional_1/dense_4/Tensordot/concat_1ConcatV20functional_1/dense_4/Tensordot/GatherV2:output:0/functional_1/dense_4/Tensordot/Const_2:output:05functional_1/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'functional_1/dense_4/Tensordot/concat_1�
functional_1/dense_4/TensordotReshape/functional_1/dense_4/Tensordot/MatMul:product:00functional_1/dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#2 
functional_1/dense_4/Tensordot�
+functional_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype02-
+functional_1/dense_4/BiasAdd/ReadVariableOp�
functional_1/dense_4/BiasAddBiasAdd'functional_1/dense_4/Tensordot:output:03functional_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#2
functional_1/dense_4/BiasAdd�
functional_1/dense_4/ReluRelu%functional_1/dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:���������#2
functional_1/dense_4/Relu�
-functional_1/dense_5/Tensordot/ReadVariableOpReadVariableOp6functional_1_dense_5_tensordot_readvariableop_resource*
_output_shapes

:#(*
dtype02/
-functional_1/dense_5/Tensordot/ReadVariableOp�
#functional_1/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#functional_1/dense_5/Tensordot/axes�
#functional_1/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#functional_1/dense_5/Tensordot/free�
$functional_1/dense_5/Tensordot/ShapeShape'functional_1/dense_4/Relu:activations:0*
T0*
_output_shapes
:2&
$functional_1/dense_5/Tensordot/Shape�
,functional_1/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense_5/Tensordot/GatherV2/axis�
'functional_1/dense_5/Tensordot/GatherV2GatherV2-functional_1/dense_5/Tensordot/Shape:output:0,functional_1/dense_5/Tensordot/free:output:05functional_1/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'functional_1/dense_5/Tensordot/GatherV2�
.functional_1/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.functional_1/dense_5/Tensordot/GatherV2_1/axis�
)functional_1/dense_5/Tensordot/GatherV2_1GatherV2-functional_1/dense_5/Tensordot/Shape:output:0,functional_1/dense_5/Tensordot/axes:output:07functional_1/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)functional_1/dense_5/Tensordot/GatherV2_1�
$functional_1/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$functional_1/dense_5/Tensordot/Const�
#functional_1/dense_5/Tensordot/ProdProd0functional_1/dense_5/Tensordot/GatherV2:output:0-functional_1/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#functional_1/dense_5/Tensordot/Prod�
&functional_1/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&functional_1/dense_5/Tensordot/Const_1�
%functional_1/dense_5/Tensordot/Prod_1Prod2functional_1/dense_5/Tensordot/GatherV2_1:output:0/functional_1/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%functional_1/dense_5/Tensordot/Prod_1�
*functional_1/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_1/dense_5/Tensordot/concat/axis�
%functional_1/dense_5/Tensordot/concatConcatV2,functional_1/dense_5/Tensordot/free:output:0,functional_1/dense_5/Tensordot/axes:output:03functional_1/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%functional_1/dense_5/Tensordot/concat�
$functional_1/dense_5/Tensordot/stackPack,functional_1/dense_5/Tensordot/Prod:output:0.functional_1/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$functional_1/dense_5/Tensordot/stack�
(functional_1/dense_5/Tensordot/transpose	Transpose'functional_1/dense_4/Relu:activations:0.functional_1/dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#2*
(functional_1/dense_5/Tensordot/transpose�
&functional_1/dense_5/Tensordot/ReshapeReshape,functional_1/dense_5/Tensordot/transpose:y:0-functional_1/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2(
&functional_1/dense_5/Tensordot/Reshape�
%functional_1/dense_5/Tensordot/MatMulMatMul/functional_1/dense_5/Tensordot/Reshape:output:05functional_1/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2'
%functional_1/dense_5/Tensordot/MatMul�
&functional_1/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2(
&functional_1/dense_5/Tensordot/Const_2�
,functional_1/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense_5/Tensordot/concat_1/axis�
'functional_1/dense_5/Tensordot/concat_1ConcatV20functional_1/dense_5/Tensordot/GatherV2:output:0/functional_1/dense_5/Tensordot/Const_2:output:05functional_1/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'functional_1/dense_5/Tensordot/concat_1�
functional_1/dense_5/TensordotReshape/functional_1/dense_5/Tensordot/MatMul:product:00functional_1/dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������(2 
functional_1/dense_5/Tensordot�
+functional_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02-
+functional_1/dense_5/BiasAdd/ReadVariableOp�
functional_1/dense_5/BiasAddBiasAdd'functional_1/dense_5/Tensordot:output:03functional_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2
functional_1/dense_5/BiasAdd�
functional_1/dense_5/ReluRelu%functional_1/dense_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������(2
functional_1/dense_5/Relu�
-functional_1/dense_6/Tensordot/ReadVariableOpReadVariableOp6functional_1_dense_6_tensordot_readvariableop_resource*
_output_shapes

:(-*
dtype02/
-functional_1/dense_6/Tensordot/ReadVariableOp�
#functional_1/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#functional_1/dense_6/Tensordot/axes�
#functional_1/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#functional_1/dense_6/Tensordot/free�
$functional_1/dense_6/Tensordot/ShapeShape'functional_1/dense_5/Relu:activations:0*
T0*
_output_shapes
:2&
$functional_1/dense_6/Tensordot/Shape�
,functional_1/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense_6/Tensordot/GatherV2/axis�
'functional_1/dense_6/Tensordot/GatherV2GatherV2-functional_1/dense_6/Tensordot/Shape:output:0,functional_1/dense_6/Tensordot/free:output:05functional_1/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'functional_1/dense_6/Tensordot/GatherV2�
.functional_1/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.functional_1/dense_6/Tensordot/GatherV2_1/axis�
)functional_1/dense_6/Tensordot/GatherV2_1GatherV2-functional_1/dense_6/Tensordot/Shape:output:0,functional_1/dense_6/Tensordot/axes:output:07functional_1/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)functional_1/dense_6/Tensordot/GatherV2_1�
$functional_1/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$functional_1/dense_6/Tensordot/Const�
#functional_1/dense_6/Tensordot/ProdProd0functional_1/dense_6/Tensordot/GatherV2:output:0-functional_1/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#functional_1/dense_6/Tensordot/Prod�
&functional_1/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&functional_1/dense_6/Tensordot/Const_1�
%functional_1/dense_6/Tensordot/Prod_1Prod2functional_1/dense_6/Tensordot/GatherV2_1:output:0/functional_1/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%functional_1/dense_6/Tensordot/Prod_1�
*functional_1/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_1/dense_6/Tensordot/concat/axis�
%functional_1/dense_6/Tensordot/concatConcatV2,functional_1/dense_6/Tensordot/free:output:0,functional_1/dense_6/Tensordot/axes:output:03functional_1/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%functional_1/dense_6/Tensordot/concat�
$functional_1/dense_6/Tensordot/stackPack,functional_1/dense_6/Tensordot/Prod:output:0.functional_1/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$functional_1/dense_6/Tensordot/stack�
(functional_1/dense_6/Tensordot/transpose	Transpose'functional_1/dense_5/Relu:activations:0.functional_1/dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������(2*
(functional_1/dense_6/Tensordot/transpose�
&functional_1/dense_6/Tensordot/ReshapeReshape,functional_1/dense_6/Tensordot/transpose:y:0-functional_1/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2(
&functional_1/dense_6/Tensordot/Reshape�
%functional_1/dense_6/Tensordot/MatMulMatMul/functional_1/dense_6/Tensordot/Reshape:output:05functional_1/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-2'
%functional_1/dense_6/Tensordot/MatMul�
&functional_1/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:-2(
&functional_1/dense_6/Tensordot/Const_2�
,functional_1/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense_6/Tensordot/concat_1/axis�
'functional_1/dense_6/Tensordot/concat_1ConcatV20functional_1/dense_6/Tensordot/GatherV2:output:0/functional_1/dense_6/Tensordot/Const_2:output:05functional_1/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'functional_1/dense_6/Tensordot/concat_1�
functional_1/dense_6/TensordotReshape/functional_1/dense_6/Tensordot/MatMul:product:00functional_1/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������-2 
functional_1/dense_6/Tensordot�
+functional_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype02-
+functional_1/dense_6/BiasAdd/ReadVariableOp�
functional_1/dense_6/BiasAddBiasAdd'functional_1/dense_6/Tensordot:output:03functional_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������-2
functional_1/dense_6/BiasAdd�
functional_1/dense_6/ReluRelu%functional_1/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:���������-2
functional_1/dense_6/Relu�
-functional_1/dense_7/Tensordot/ReadVariableOpReadVariableOp6functional_1_dense_7_tensordot_readvariableop_resource*
_output_shapes

:-7*
dtype02/
-functional_1/dense_7/Tensordot/ReadVariableOp�
#functional_1/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#functional_1/dense_7/Tensordot/axes�
#functional_1/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#functional_1/dense_7/Tensordot/free�
$functional_1/dense_7/Tensordot/ShapeShape'functional_1/dense_6/Relu:activations:0*
T0*
_output_shapes
:2&
$functional_1/dense_7/Tensordot/Shape�
,functional_1/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense_7/Tensordot/GatherV2/axis�
'functional_1/dense_7/Tensordot/GatherV2GatherV2-functional_1/dense_7/Tensordot/Shape:output:0,functional_1/dense_7/Tensordot/free:output:05functional_1/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'functional_1/dense_7/Tensordot/GatherV2�
.functional_1/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.functional_1/dense_7/Tensordot/GatherV2_1/axis�
)functional_1/dense_7/Tensordot/GatherV2_1GatherV2-functional_1/dense_7/Tensordot/Shape:output:0,functional_1/dense_7/Tensordot/axes:output:07functional_1/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)functional_1/dense_7/Tensordot/GatherV2_1�
$functional_1/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$functional_1/dense_7/Tensordot/Const�
#functional_1/dense_7/Tensordot/ProdProd0functional_1/dense_7/Tensordot/GatherV2:output:0-functional_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#functional_1/dense_7/Tensordot/Prod�
&functional_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&functional_1/dense_7/Tensordot/Const_1�
%functional_1/dense_7/Tensordot/Prod_1Prod2functional_1/dense_7/Tensordot/GatherV2_1:output:0/functional_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%functional_1/dense_7/Tensordot/Prod_1�
*functional_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_1/dense_7/Tensordot/concat/axis�
%functional_1/dense_7/Tensordot/concatConcatV2,functional_1/dense_7/Tensordot/free:output:0,functional_1/dense_7/Tensordot/axes:output:03functional_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%functional_1/dense_7/Tensordot/concat�
$functional_1/dense_7/Tensordot/stackPack,functional_1/dense_7/Tensordot/Prod:output:0.functional_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$functional_1/dense_7/Tensordot/stack�
(functional_1/dense_7/Tensordot/transpose	Transpose'functional_1/dense_6/Relu:activations:0.functional_1/dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������-2*
(functional_1/dense_7/Tensordot/transpose�
&functional_1/dense_7/Tensordot/ReshapeReshape,functional_1/dense_7/Tensordot/transpose:y:0-functional_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2(
&functional_1/dense_7/Tensordot/Reshape�
%functional_1/dense_7/Tensordot/MatMulMatMul/functional_1/dense_7/Tensordot/Reshape:output:05functional_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������72'
%functional_1/dense_7/Tensordot/MatMul�
&functional_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:72(
&functional_1/dense_7/Tensordot/Const_2�
,functional_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_1/dense_7/Tensordot/concat_1/axis�
'functional_1/dense_7/Tensordot/concat_1ConcatV20functional_1/dense_7/Tensordot/GatherV2:output:0/functional_1/dense_7/Tensordot/Const_2:output:05functional_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'functional_1/dense_7/Tensordot/concat_1�
functional_1/dense_7/TensordotReshape/functional_1/dense_7/Tensordot/MatMul:product:00functional_1/dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������72 
functional_1/dense_7/Tensordot�
+functional_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype02-
+functional_1/dense_7/BiasAdd/ReadVariableOp�
functional_1/dense_7/BiasAddBiasAdd'functional_1/dense_7/Tensordot:output:03functional_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������72
functional_1/dense_7/BiasAdd�
functional_1/dense_7/SigmoidSigmoid%functional_1/dense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������72
functional_1/dense_7/Sigmoidx
IdentityIdentity functional_1/dense_7/Sigmoid:y:0*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������7:::::::::::::::::T P
+
_output_shapes
:���������7
!
_user_specified_name	input_1
�
|
'__inference_dense_5_layer_call_fn_90239

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_891202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������#::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������#
 
_user_specified_nameinputs
�)
�
G__inference_functional_1_layer_call_and_return_conditional_losses_89275
input_1
dense_89234
dense_89236
dense_1_89239
dense_1_89241
dense_2_89244
dense_2_89246
dense_3_89249
dense_3_89251
dense_4_89254
dense_4_89256
dense_5_89259
dense_5_89261
dense_6_89264
dense_6_89266
dense_7_89269
dense_7_89271
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_89234dense_89236*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_888852
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_89239dense_1_89241*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_889322!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_89244dense_2_89246*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_889792!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_89249dense_3_89251*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_890262!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_89254dense_4_89256*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_890732!
dense_4/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_89259dense_5_89261*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_891202!
dense_5/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_89264dense_6_89266*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_891672!
dense_6/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_89269dense_7_89271*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������7*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_892142!
dense_7/StatefulPartitionedCall�
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������7::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:T P
+
_output_shapes
:���������7
!
_user_specified_name	input_1
�)
�
G__inference_functional_1_layer_call_and_return_conditional_losses_89231
input_1
dense_88896
dense_88898
dense_1_88943
dense_1_88945
dense_2_88990
dense_2_88992
dense_3_89037
dense_3_89039
dense_4_89084
dense_4_89086
dense_5_89131
dense_5_89133
dense_6_89178
dense_6_89180
dense_7_89225
dense_7_89227
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_88896dense_88898*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_888852
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_88943dense_1_88945*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_889322!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_88990dense_2_88992*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_889792!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_89037dense_3_89039*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_890262!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_89084dense_4_89086*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_890732!
dense_4/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_89131dense_5_89133*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_891202!
dense_5/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_89178dense_6_89180*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_891672!
dense_6/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_89225dense_7_89227*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������7*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_892142!
dense_7/StatefulPartitionedCall�
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������7::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:T P
+
_output_shapes
:���������7
!
_user_specified_name	input_1
�
|
'__inference_dense_4_layer_call_fn_90199

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_890732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������#2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_dense_6_layer_call_and_return_conditional_losses_90270

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:(-*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������(2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:-2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������-2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������-2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������-2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������-2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������(:::S O
+
_output_shapes
:���������(
 
_user_specified_nameinputs
�
|
'__inference_dense_2_layer_call_fn_90119

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_889792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������#2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������(::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������(
 
_user_specified_nameinputs
�
|
'__inference_dense_7_layer_call_fn_90319

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������7*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_892142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������-::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������-
 
_user_specified_nameinputs
�
|
'__inference_dense_6_layer_call_fn_90279

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_891672
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������-2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������(::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
@__inference_dense_layer_call_and_return_conditional_losses_90030

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:72*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������72
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:22
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������22
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������22	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������22
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������7:::S O
+
_output_shapes
:���������7
 
_user_specified_nameinputs
�p
�
__inference__traced_save_90513
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_6a766506b3064aff9b34f1960a7ad854/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename� 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :72:2:2(:(:(#:#:	#�:�:	�#:#:#(:(:(-:-:-7:7: : : : : : : : : :72:2:2(:(:(#:#:	#�:�:	�#:#:#(:(:(-:-:-7:7:72:2:2(:(:(#:#:	#�:�:	�#:#:#(:(:(-:-:-7:7: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:72: 

_output_shapes
:2:$ 

_output_shapes

:2(: 

_output_shapes
:(:$ 

_output_shapes

:(#: 

_output_shapes
:#:%!

_output_shapes
:	#�:!

_output_shapes	
:�:%	!

_output_shapes
:	�#: 


_output_shapes
:#:$ 

_output_shapes

:#(: 

_output_shapes
:(:$ 

_output_shapes

:(-: 

_output_shapes
:-:$ 

_output_shapes

:-7: 

_output_shapes
:7:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:72: 

_output_shapes
:2:$ 

_output_shapes

:2(: 

_output_shapes
:(:$ 

_output_shapes

:(#: 

_output_shapes
:#:% !

_output_shapes
:	#�:!!

_output_shapes	
:�:%"!

_output_shapes
:	�#: #

_output_shapes
:#:$$ 

_output_shapes

:#(: %

_output_shapes
:(:$& 

_output_shapes

:(-: '

_output_shapes
:-:$( 

_output_shapes

:-7: )

_output_shapes
:7:$* 

_output_shapes

:72: +

_output_shapes
:2:$, 

_output_shapes

:2(: -

_output_shapes
:(:$. 

_output_shapes

:(#: /

_output_shapes
:#:%0!

_output_shapes
:	#�:!1

_output_shapes	
:�:%2!

_output_shapes
:	�#: 3

_output_shapes
:#:$4 

_output_shapes

:#(: 5

_output_shapes
:(:$6 

_output_shapes

:(-: 7

_output_shapes
:-:$8 

_output_shapes

:-7: 9

_output_shapes
:7::

_output_shapes
: 
�
�
B__inference_dense_2_layer_call_and_return_conditional_losses_88979

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:(#*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������(2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:#*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������#2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������#2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������(:::S O
+
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
B__inference_dense_7_layer_call_and_return_conditional_losses_89214

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:-7*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������-2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������72
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:72
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������72
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:7*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������72	
BiasAdde
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:���������72	
Sigmoidc
IdentityIdentitySigmoid:y:0*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������-:::S O
+
_output_shapes
:���������-
 
_user_specified_nameinputs
�
|
'__inference_dense_1_layer_call_fn_90079

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_889322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������2::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������2
 
_user_specified_nameinputs
��
�
G__inference_functional_1_layer_call_and_return_conditional_losses_89925

inputs+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource-
)dense_3_tensordot_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource-
)dense_4_tensordot_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource-
)dense_5_tensordot_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource-
)dense_6_tensordot_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource-
)dense_7_tensordot_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity��
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:72*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freed
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense/Tensordot/Shape�
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis�
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2�
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis�
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const�
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1�
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis�
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack�
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������72
dense/Tensordot/transpose�
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense/Tensordot/Reshape�
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:22
dense/Tensordot/Const_2�
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis�
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������22
dense/Tensordot�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������22
dense/BiasAddn

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:���������22

dense/Relu�
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:2(*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axes�
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/freez
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dense_1/Tensordot/Shape�
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axis�
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2�
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis�
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2_1|
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const�
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod�
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1�
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1�
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axis�
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat�
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stack�
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������22
dense_1/Tensordot/transpose�
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_1/Tensordot/Reshape�
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_1/Tensordot/MatMul�
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2
dense_1/Tensordot/Const_2�
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axis�
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1�
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������(2
dense_1/Tensordot�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2
dense_1/BiasAddt
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������(2
dense_1/Relu�
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:(#*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes�
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/free|
dense_2/Tensordot/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape�
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis�
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2�
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis�
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const�
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod�
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1�
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1�
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis�
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat�
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack�
dense_2/Tensordot/transpose	Transposedense_1/Relu:activations:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������(2
dense_2/Tensordot/transpose�
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_2/Tensordot/Reshape�
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#2
dense_2/Tensordot/MatMul�
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2
dense_2/Tensordot/Const_2�
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis�
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1�
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#2
dense_2/Tensordot�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#2
dense_2/BiasAddt
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������#2
dense_2/Relu�
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes
:	#�*
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axes�
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_3/Tensordot/free|
dense_3/Tensordot/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
dense_3/Tensordot/Shape�
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axis�
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2�
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis�
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2_1|
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const�
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod�
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1�
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1�
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axis�
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat�
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stack�
dense_3/Tensordot/transpose	Transposedense_2/Relu:activations:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#2
dense_3/Tensordot/transpose�
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_3/Tensordot/Reshape�
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_3/Tensordot/MatMul�
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
dense_3/Tensordot/Const_2�
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axis�
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1�
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
dense_3/Tensordot�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
dense_3/BiasAddu
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
dense_3/Relu�
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	�#*
dtype02"
 dense_4/Tensordot/ReadVariableOpz
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/axes�
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_4/Tensordot/free|
dense_4/Tensordot/ShapeShapedense_3/Relu:activations:0*
T0*
_output_shapes
:2
dense_4/Tensordot/Shape�
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/GatherV2/axis�
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2�
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_4/Tensordot/GatherV2_1/axis�
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2_1|
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const�
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod�
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const_1�
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod_1�
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_4/Tensordot/concat/axis�
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat�
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/stack�
dense_4/Tensordot/transpose	Transposedense_3/Relu:activations:0!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������2
dense_4/Tensordot/transpose�
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_4/Tensordot/Reshape�
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#2
dense_4/Tensordot/MatMul�
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2
dense_4/Tensordot/Const_2�
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/concat_1/axis�
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat_1�
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#2
dense_4/Tensordot�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#2
dense_4/BiasAddt
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*+
_output_shapes
:���������#2
dense_4/Relu�
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes

:#(*
dtype02"
 dense_5/Tensordot/ReadVariableOpz
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/axes�
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_5/Tensordot/free|
dense_5/Tensordot/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:2
dense_5/Tensordot/Shape�
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/GatherV2/axis�
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2�
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_5/Tensordot/GatherV2_1/axis�
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2_1|
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const�
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod�
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const_1�
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod_1�
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_5/Tensordot/concat/axis�
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat�
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/stack�
dense_5/Tensordot/transpose	Transposedense_4/Relu:activations:0!dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#2
dense_5/Tensordot/transpose�
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_5/Tensordot/Reshape�
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_5/Tensordot/MatMul�
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2
dense_5/Tensordot/Const_2�
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/concat_1/axis�
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat_1�
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������(2
dense_5/Tensordot�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02 
dense_5/BiasAdd/ReadVariableOp�
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2
dense_5/BiasAddt
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������(2
dense_5/Relu�
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes

:(-*
dtype02"
 dense_6/Tensordot/ReadVariableOpz
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/axes�
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_6/Tensordot/free|
dense_6/Tensordot/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:2
dense_6/Tensordot/Shape�
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/GatherV2/axis�
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2�
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_6/Tensordot/GatherV2_1/axis�
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2_1|
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const�
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod�
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const_1�
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod_1�
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_6/Tensordot/concat/axis�
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat�
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/stack�
dense_6/Tensordot/transpose	Transposedense_5/Relu:activations:0!dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������(2
dense_6/Tensordot/transpose�
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_6/Tensordot/Reshape�
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-2
dense_6/Tensordot/MatMul�
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:-2
dense_6/Tensordot/Const_2�
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/concat_1/axis�
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat_1�
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������-2
dense_6/Tensordot�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype02 
dense_6/BiasAdd/ReadVariableOp�
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������-2
dense_6/BiasAddt
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*+
_output_shapes
:���������-2
dense_6/Relu�
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes

:-7*
dtype02"
 dense_7/Tensordot/ReadVariableOpz
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/axes�
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_7/Tensordot/free|
dense_7/Tensordot/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:2
dense_7/Tensordot/Shape�
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/GatherV2/axis�
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2�
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_7/Tensordot/GatherV2_1/axis�
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2_1|
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const�
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod�
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const_1�
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod_1�
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_7/Tensordot/concat/axis�
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat�
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/stack�
dense_7/Tensordot/transpose	Transposedense_6/Relu:activations:0!dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������-2
dense_7/Tensordot/transpose�
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_7/Tensordot/Reshape�
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������72
dense_7/Tensordot/MatMul�
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:72
dense_7/Tensordot/Const_2�
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/concat_1/axis�
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat_1�
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������72
dense_7/Tensordot�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype02 
dense_7/BiasAdd/ReadVariableOp�
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������72
dense_7/BiasAdd}
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������72
dense_7/Sigmoidk
IdentityIdentitydense_7/Sigmoid:y:0*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������7:::::::::::::::::S O
+
_output_shapes
:���������7
 
_user_specified_nameinputs
�

�
,__inference_functional_1_layer_call_fn_89438
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������7*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_894032
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������7::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������7
!
_user_specified_name	input_1
�

�
,__inference_functional_1_layer_call_fn_89357
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������7*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_893222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������7::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������7
!
_user_specified_name	input_1
�
�
B__inference_dense_7_layer_call_and_return_conditional_losses_90310

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:-7*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������-2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������72
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:72
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������72
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:7*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������72	
BiasAdde
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:���������72	
Sigmoidc
IdentityIdentitySigmoid:y:0*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������-:::S O
+
_output_shapes
:���������-
 
_user_specified_nameinputs
�

�
,__inference_functional_1_layer_call_fn_89962

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������7*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_893222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������7::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������7
 
_user_specified_nameinputs
�
�
@__inference_dense_layer_call_and_return_conditional_losses_88885

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:72*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������72
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:22
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������22
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������22	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������22
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������7:::S O
+
_output_shapes
:���������7
 
_user_specified_nameinputs
�
�
B__inference_dense_5_layer_call_and_return_conditional_losses_90230

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:#(*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������#2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������(2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������(2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������#:::S O
+
_output_shapes
:���������#
 
_user_specified_nameinputs
�
z
%__inference_dense_layer_call_fn_90039

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_888852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������7::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������7
 
_user_specified_nameinputs
�

�
#__inference_signature_wrapper_89485
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������7*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_888502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������7::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������7
!
_user_specified_name	input_1
�
�
B__inference_dense_4_layer_call_and_return_conditional_losses_89073

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�#*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:����������2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:#*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������#2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������#2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������:::T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
!__inference__traced_restore_90694
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias%
!assignvariableop_6_dense_3_kernel#
assignvariableop_7_dense_3_bias%
!assignvariableop_8_dense_4_kernel#
assignvariableop_9_dense_4_bias&
"assignvariableop_10_dense_5_kernel$
 assignvariableop_11_dense_5_bias&
"assignvariableop_12_dense_6_kernel$
 assignvariableop_13_dense_6_bias&
"assignvariableop_14_dense_7_kernel$
 assignvariableop_15_dense_7_bias!
assignvariableop_16_adam_iter#
assignvariableop_17_adam_beta_1#
assignvariableop_18_adam_beta_2"
assignvariableop_19_adam_decay*
&assignvariableop_20_adam_learning_rate
assignvariableop_21_total
assignvariableop_22_count
assignvariableop_23_total_1
assignvariableop_24_count_1+
'assignvariableop_25_adam_dense_kernel_m)
%assignvariableop_26_adam_dense_bias_m-
)assignvariableop_27_adam_dense_1_kernel_m+
'assignvariableop_28_adam_dense_1_bias_m-
)assignvariableop_29_adam_dense_2_kernel_m+
'assignvariableop_30_adam_dense_2_bias_m-
)assignvariableop_31_adam_dense_3_kernel_m+
'assignvariableop_32_adam_dense_3_bias_m-
)assignvariableop_33_adam_dense_4_kernel_m+
'assignvariableop_34_adam_dense_4_bias_m-
)assignvariableop_35_adam_dense_5_kernel_m+
'assignvariableop_36_adam_dense_5_bias_m-
)assignvariableop_37_adam_dense_6_kernel_m+
'assignvariableop_38_adam_dense_6_bias_m-
)assignvariableop_39_adam_dense_7_kernel_m+
'assignvariableop_40_adam_dense_7_bias_m+
'assignvariableop_41_adam_dense_kernel_v)
%assignvariableop_42_adam_dense_bias_v-
)assignvariableop_43_adam_dense_1_kernel_v+
'assignvariableop_44_adam_dense_1_bias_v-
)assignvariableop_45_adam_dense_2_kernel_v+
'assignvariableop_46_adam_dense_2_bias_v-
)assignvariableop_47_adam_dense_3_kernel_v+
'assignvariableop_48_adam_dense_3_bias_v-
)assignvariableop_49_adam_dense_4_kernel_v+
'assignvariableop_50_adam_dense_4_bias_v-
)assignvariableop_51_adam_dense_5_kernel_v+
'assignvariableop_52_adam_dense_5_bias_v-
)assignvariableop_53_adam_dense_6_kernel_v+
'assignvariableop_54_adam_dense_6_bias_v-
)assignvariableop_55_adam_dense_7_kernel_v+
'assignvariableop_56_adam_dense_7_bias_v
identity_58��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9� 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_2_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_2_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_3_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_3_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_4_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_4_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_5_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_5_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_6_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_6_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_7_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_7_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_dense_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp%assignvariableop_42_adam_dense_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_1_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_1_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_2_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_2_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_3_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_3_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_4_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_4_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_5_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_5_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_dense_6_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_6_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_dense_7_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_7_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_569
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_57�

Identity_58IdentityIdentity_57:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_58"#
identity_58Identity_58:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
,__inference_functional_1_layer_call_fn_89999

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������7*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_894032
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������7::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������7
 
_user_specified_nameinputs
�
�
B__inference_dense_1_layer_call_and_return_conditional_losses_90070

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:2(*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������22
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������(2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������(2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������2:::S O
+
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
B__inference_dense_2_layer_call_and_return_conditional_losses_90110

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:(#*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������(2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:#*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������#2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������#2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������(:::S O
+
_output_shapes
:���������(
 
_user_specified_nameinputs
��
�
G__inference_functional_1_layer_call_and_return_conditional_losses_89705

inputs+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource-
)dense_3_tensordot_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource-
)dense_4_tensordot_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource-
)dense_5_tensordot_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource-
)dense_6_tensordot_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource-
)dense_7_tensordot_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity��
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:72*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freed
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense/Tensordot/Shape�
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis�
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2�
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis�
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const�
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1�
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis�
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack�
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������72
dense/Tensordot/transpose�
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense/Tensordot/Reshape�
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:22
dense/Tensordot/Const_2�
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis�
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������22
dense/Tensordot�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������22
dense/BiasAddn

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:���������22

dense/Relu�
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:2(*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axes�
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/freez
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dense_1/Tensordot/Shape�
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axis�
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2�
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis�
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2_1|
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const�
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod�
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1�
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1�
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axis�
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat�
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stack�
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������22
dense_1/Tensordot/transpose�
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_1/Tensordot/Reshape�
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_1/Tensordot/MatMul�
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2
dense_1/Tensordot/Const_2�
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axis�
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1�
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������(2
dense_1/Tensordot�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2
dense_1/BiasAddt
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������(2
dense_1/Relu�
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:(#*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes�
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/free|
dense_2/Tensordot/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape�
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis�
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2�
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis�
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const�
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod�
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1�
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1�
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis�
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat�
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack�
dense_2/Tensordot/transpose	Transposedense_1/Relu:activations:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������(2
dense_2/Tensordot/transpose�
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_2/Tensordot/Reshape�
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#2
dense_2/Tensordot/MatMul�
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2
dense_2/Tensordot/Const_2�
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis�
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1�
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#2
dense_2/Tensordot�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#2
dense_2/BiasAddt
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������#2
dense_2/Relu�
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes
:	#�*
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axes�
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_3/Tensordot/free|
dense_3/Tensordot/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
dense_3/Tensordot/Shape�
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axis�
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2�
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis�
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2_1|
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const�
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod�
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1�
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1�
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axis�
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat�
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stack�
dense_3/Tensordot/transpose	Transposedense_2/Relu:activations:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#2
dense_3/Tensordot/transpose�
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_3/Tensordot/Reshape�
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_3/Tensordot/MatMul�
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
dense_3/Tensordot/Const_2�
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axis�
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1�
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
dense_3/Tensordot�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
dense_3/BiasAddu
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
dense_3/Relu�
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	�#*
dtype02"
 dense_4/Tensordot/ReadVariableOpz
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/axes�
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_4/Tensordot/free|
dense_4/Tensordot/ShapeShapedense_3/Relu:activations:0*
T0*
_output_shapes
:2
dense_4/Tensordot/Shape�
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/GatherV2/axis�
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2�
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_4/Tensordot/GatherV2_1/axis�
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2_1|
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const�
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod�
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const_1�
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod_1�
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_4/Tensordot/concat/axis�
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat�
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/stack�
dense_4/Tensordot/transpose	Transposedense_3/Relu:activations:0!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������2
dense_4/Tensordot/transpose�
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_4/Tensordot/Reshape�
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#2
dense_4/Tensordot/MatMul�
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:#2
dense_4/Tensordot/Const_2�
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/concat_1/axis�
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat_1�
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#2
dense_4/Tensordot�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#2
dense_4/BiasAddt
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*+
_output_shapes
:���������#2
dense_4/Relu�
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes

:#(*
dtype02"
 dense_5/Tensordot/ReadVariableOpz
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/axes�
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_5/Tensordot/free|
dense_5/Tensordot/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:2
dense_5/Tensordot/Shape�
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/GatherV2/axis�
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2�
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_5/Tensordot/GatherV2_1/axis�
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2_1|
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const�
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod�
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const_1�
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod_1�
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_5/Tensordot/concat/axis�
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat�
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/stack�
dense_5/Tensordot/transpose	Transposedense_4/Relu:activations:0!dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#2
dense_5/Tensordot/transpose�
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_5/Tensordot/Reshape�
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_5/Tensordot/MatMul�
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2
dense_5/Tensordot/Const_2�
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/concat_1/axis�
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat_1�
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������(2
dense_5/Tensordot�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02 
dense_5/BiasAdd/ReadVariableOp�
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2
dense_5/BiasAddt
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������(2
dense_5/Relu�
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes

:(-*
dtype02"
 dense_6/Tensordot/ReadVariableOpz
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/axes�
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_6/Tensordot/free|
dense_6/Tensordot/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:2
dense_6/Tensordot/Shape�
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/GatherV2/axis�
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2�
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_6/Tensordot/GatherV2_1/axis�
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2_1|
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const�
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod�
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const_1�
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod_1�
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_6/Tensordot/concat/axis�
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat�
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/stack�
dense_6/Tensordot/transpose	Transposedense_5/Relu:activations:0!dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������(2
dense_6/Tensordot/transpose�
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_6/Tensordot/Reshape�
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-2
dense_6/Tensordot/MatMul�
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:-2
dense_6/Tensordot/Const_2�
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/concat_1/axis�
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat_1�
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������-2
dense_6/Tensordot�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype02 
dense_6/BiasAdd/ReadVariableOp�
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������-2
dense_6/BiasAddt
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*+
_output_shapes
:���������-2
dense_6/Relu�
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes

:-7*
dtype02"
 dense_7/Tensordot/ReadVariableOpz
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/axes�
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_7/Tensordot/free|
dense_7/Tensordot/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:2
dense_7/Tensordot/Shape�
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/GatherV2/axis�
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2�
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_7/Tensordot/GatherV2_1/axis�
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2_1|
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const�
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod�
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const_1�
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod_1�
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_7/Tensordot/concat/axis�
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat�
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/stack�
dense_7/Tensordot/transpose	Transposedense_6/Relu:activations:0!dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������-2
dense_7/Tensordot/transpose�
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_7/Tensordot/Reshape�
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������72
dense_7/Tensordot/MatMul�
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:72
dense_7/Tensordot/Const_2�
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/concat_1/axis�
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat_1�
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������72
dense_7/Tensordot�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype02 
dense_7/BiasAdd/ReadVariableOp�
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������72
dense_7/BiasAdd}
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*+
_output_shapes
:���������72
dense_7/Sigmoidk
IdentityIdentitydense_7/Sigmoid:y:0*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������7:::::::::::::::::S O
+
_output_shapes
:���������7
 
_user_specified_nameinputs
�
�
B__inference_dense_3_layer_call_and_return_conditional_losses_89026

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	#�*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������#2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������#:::S O
+
_output_shapes
:���������#
 
_user_specified_nameinputs
�
�
B__inference_dense_3_layer_call_and_return_conditional_losses_90150

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	#�*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������#2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������#:::S O
+
_output_shapes
:���������#
 
_user_specified_nameinputs
�
�
B__inference_dense_1_layer_call_and_return_conditional_losses_88932

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:2(*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������22
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������(2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������(2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������(2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������2:::S O
+
_output_shapes
:���������2
 
_user_specified_nameinputs
�)
�
G__inference_functional_1_layer_call_and_return_conditional_losses_89403

inputs
dense_89362
dense_89364
dense_1_89367
dense_1_89369
dense_2_89372
dense_2_89374
dense_3_89377
dense_3_89379
dense_4_89382
dense_4_89384
dense_5_89387
dense_5_89389
dense_6_89392
dense_6_89394
dense_7_89397
dense_7_89399
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_89362dense_89364*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_888852
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_89367dense_1_89369*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_889322!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_89372dense_2_89374*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_889792!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_89377dense_3_89379*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_890262!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_89382dense_4_89384*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_890732!
dense_4/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_89387dense_5_89389*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_891202!
dense_5/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_89392dense_6_89394*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_891672!
dense_6/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_89397dense_7_89399*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������7*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_892142!
dense_7/StatefulPartitionedCall�
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*+
_output_shapes
:���������72

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������7::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:S O
+
_output_shapes
:���������7
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������7?
dense_74
StatefulPartitionedCall:0���������7tensorflow/serving/predict:ϙ
�L
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
�_default_save_signature
+�&call_and_return_all_conditional_losses
�__call__"�H
_tf_keras_network�H{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 55]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1000, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 45, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 55, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_7", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 55]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 55]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1000, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 45, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 55, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_7", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 55]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 55]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 55}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 55]}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 50]}}
�

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 40]}}
�

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1000, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 35}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 35]}}
�

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 1000]}}
�

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 35}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 35]}}
�

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 45, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 40]}}
�

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 55, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 45}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 45]}}
�
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_ratem}m~mm�m�m�"m�#m�(m�)m�.m�/m�4m�5m�:m�;m�v�v�v�v�v�v�"v�#v�(v�)v�.v�/v�4v�5v�:v�;v�"
	optimizer
 "
trackable_list_wrapper
�
0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15"
trackable_list_wrapper
�
0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15"
trackable_list_wrapper
�
Emetrics
regularization_losses
	variables
Fnon_trainable_variables

Glayers
trainable_variables
Hlayer_metrics
Ilayer_regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
:722dense/kernel
:22
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Jmetrics
regularization_losses
	variables
Knon_trainable_variables

Llayers
trainable_variables
Mlayer_metrics
Nlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :2(2dense_1/kernel
:(2dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Ometrics
regularization_losses
	variables
Pnon_trainable_variables

Qlayers
trainable_variables
Rlayer_metrics
Slayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :(#2dense_2/kernel
:#2dense_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Tmetrics
regularization_losses
	variables
Unon_trainable_variables

Vlayers
 trainable_variables
Wlayer_metrics
Xlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	#�2dense_3/kernel
:�2dense_3/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
�
Ymetrics
$regularization_losses
%	variables
Znon_trainable_variables

[layers
&trainable_variables
\layer_metrics
]layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	�#2dense_4/kernel
:#2dense_4/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
�
^metrics
*regularization_losses
+	variables
_non_trainable_variables

`layers
,trainable_variables
alayer_metrics
blayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :#(2dense_5/kernel
:(2dense_5/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
�
cmetrics
0regularization_losses
1	variables
dnon_trainable_variables

elayers
2trainable_variables
flayer_metrics
glayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :(-2dense_6/kernel
:-2dense_6/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
�
hmetrics
6regularization_losses
7	variables
inon_trainable_variables

jlayers
8trainable_variables
klayer_metrics
llayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :-72dense_7/kernel
:72dense_7/bias
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
�
mmetrics
<regularization_losses
=	variables
nnon_trainable_variables

olayers
>trainable_variables
player_metrics
qlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�
	ttotal
	ucount
v	variables
w	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	xtotal
	ycount
z
_fn_kwargs
{	variables
|	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
t0
u1"
trackable_list_wrapper
-
v	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
x0
y1"
trackable_list_wrapper
-
{	variables"
_generic_user_object
#:!722Adam/dense/kernel/m
:22Adam/dense/bias/m
%:#2(2Adam/dense_1/kernel/m
:(2Adam/dense_1/bias/m
%:#(#2Adam/dense_2/kernel/m
:#2Adam/dense_2/bias/m
&:$	#�2Adam/dense_3/kernel/m
 :�2Adam/dense_3/bias/m
&:$	�#2Adam/dense_4/kernel/m
:#2Adam/dense_4/bias/m
%:##(2Adam/dense_5/kernel/m
:(2Adam/dense_5/bias/m
%:#(-2Adam/dense_6/kernel/m
:-2Adam/dense_6/bias/m
%:#-72Adam/dense_7/kernel/m
:72Adam/dense_7/bias/m
#:!722Adam/dense/kernel/v
:22Adam/dense/bias/v
%:#2(2Adam/dense_1/kernel/v
:(2Adam/dense_1/bias/v
%:#(#2Adam/dense_2/kernel/v
:#2Adam/dense_2/bias/v
&:$	#�2Adam/dense_3/kernel/v
 :�2Adam/dense_3/bias/v
&:$	�#2Adam/dense_4/kernel/v
:#2Adam/dense_4/bias/v
%:##(2Adam/dense_5/kernel/v
:(2Adam/dense_5/bias/v
%:#(-2Adam/dense_6/kernel/v
:-2Adam/dense_6/bias/v
%:#-72Adam/dense_7/kernel/v
:72Adam/dense_7/bias/v
�2�
 __inference__wrapped_model_88850�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_1���������7
�2�
G__inference_functional_1_layer_call_and_return_conditional_losses_89705
G__inference_functional_1_layer_call_and_return_conditional_losses_89925
G__inference_functional_1_layer_call_and_return_conditional_losses_89275
G__inference_functional_1_layer_call_and_return_conditional_losses_89231�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_functional_1_layer_call_fn_89438
,__inference_functional_1_layer_call_fn_89999
,__inference_functional_1_layer_call_fn_89357
,__inference_functional_1_layer_call_fn_89962�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_dense_layer_call_and_return_conditional_losses_90030�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_dense_layer_call_fn_90039�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_1_layer_call_and_return_conditional_losses_90070�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_1_layer_call_fn_90079�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_2_layer_call_and_return_conditional_losses_90110�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_2_layer_call_fn_90119�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_3_layer_call_and_return_conditional_losses_90150�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_3_layer_call_fn_90159�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_4_layer_call_and_return_conditional_losses_90190�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_4_layer_call_fn_90199�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_5_layer_call_and_return_conditional_losses_90230�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_5_layer_call_fn_90239�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_6_layer_call_and_return_conditional_losses_90270�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_6_layer_call_fn_90279�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_7_layer_call_and_return_conditional_losses_90310�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_7_layer_call_fn_90319�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
2B0
#__inference_signature_wrapper_89485input_1�
 __inference__wrapped_model_88850"#()./45:;4�1
*�'
%�"
input_1���������7
� "5�2
0
dense_7%�"
dense_7���������7�
B__inference_dense_1_layer_call_and_return_conditional_losses_90070d3�0
)�&
$�!
inputs���������2
� ")�&
�
0���������(
� �
'__inference_dense_1_layer_call_fn_90079W3�0
)�&
$�!
inputs���������2
� "����������(�
B__inference_dense_2_layer_call_and_return_conditional_losses_90110d3�0
)�&
$�!
inputs���������(
� ")�&
�
0���������#
� �
'__inference_dense_2_layer_call_fn_90119W3�0
)�&
$�!
inputs���������(
� "����������#�
B__inference_dense_3_layer_call_and_return_conditional_losses_90150e"#3�0
)�&
$�!
inputs���������#
� "*�'
 �
0����������
� �
'__inference_dense_3_layer_call_fn_90159X"#3�0
)�&
$�!
inputs���������#
� "������������
B__inference_dense_4_layer_call_and_return_conditional_losses_90190e()4�1
*�'
%�"
inputs����������
� ")�&
�
0���������#
� �
'__inference_dense_4_layer_call_fn_90199X()4�1
*�'
%�"
inputs����������
� "����������#�
B__inference_dense_5_layer_call_and_return_conditional_losses_90230d./3�0
)�&
$�!
inputs���������#
� ")�&
�
0���������(
� �
'__inference_dense_5_layer_call_fn_90239W./3�0
)�&
$�!
inputs���������#
� "����������(�
B__inference_dense_6_layer_call_and_return_conditional_losses_90270d453�0
)�&
$�!
inputs���������(
� ")�&
�
0���������-
� �
'__inference_dense_6_layer_call_fn_90279W453�0
)�&
$�!
inputs���������(
� "����������-�
B__inference_dense_7_layer_call_and_return_conditional_losses_90310d:;3�0
)�&
$�!
inputs���������-
� ")�&
�
0���������7
� �
'__inference_dense_7_layer_call_fn_90319W:;3�0
)�&
$�!
inputs���������-
� "����������7�
@__inference_dense_layer_call_and_return_conditional_losses_90030d3�0
)�&
$�!
inputs���������7
� ")�&
�
0���������2
� �
%__inference_dense_layer_call_fn_90039W3�0
)�&
$�!
inputs���������7
� "����������2�
G__inference_functional_1_layer_call_and_return_conditional_losses_89231{"#()./45:;<�9
2�/
%�"
input_1���������7
p

 
� ")�&
�
0���������7
� �
G__inference_functional_1_layer_call_and_return_conditional_losses_89275{"#()./45:;<�9
2�/
%�"
input_1���������7
p 

 
� ")�&
�
0���������7
� �
G__inference_functional_1_layer_call_and_return_conditional_losses_89705z"#()./45:;;�8
1�.
$�!
inputs���������7
p

 
� ")�&
�
0���������7
� �
G__inference_functional_1_layer_call_and_return_conditional_losses_89925z"#()./45:;;�8
1�.
$�!
inputs���������7
p 

 
� ")�&
�
0���������7
� �
,__inference_functional_1_layer_call_fn_89357n"#()./45:;<�9
2�/
%�"
input_1���������7
p

 
� "����������7�
,__inference_functional_1_layer_call_fn_89438n"#()./45:;<�9
2�/
%�"
input_1���������7
p 

 
� "����������7�
,__inference_functional_1_layer_call_fn_89962m"#()./45:;;�8
1�.
$�!
inputs���������7
p

 
� "����������7�
,__inference_functional_1_layer_call_fn_89999m"#()./45:;;�8
1�.
$�!
inputs���������7
p 

 
� "����������7�
#__inference_signature_wrapper_89485�"#()./45:;?�<
� 
5�2
0
input_1%�"
input_1���������7"5�2
0
dense_7%�"
dense_7���������7