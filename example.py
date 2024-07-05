from value import Value
from trace import draw_dot

a = Value(2.0, label='A')
b = Value(-3.0, label='B')
c = Value(0.0, label='C')
d = Value(1.0, label='D')
bias = Value(6.8813735870, label='Bias')

ab = a * b
ab.label = 'A*B'
cd = c * d
cd.label = 'C*D'
abcd = ab + cd
abcd.label = 'A*B+C*D'
n = abcd + bias
n.label = 'N'
o = n.tanh()
o.label = 'O'

o.backward()
draw_dot(o)