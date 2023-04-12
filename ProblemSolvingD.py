import numpy as np
import matplotlib.pyplot as plt
import os
import math

# load txt files as arrays
water = np.loadtxt('pure-water.txt')
hwater = np.loadtxt('pure-heavy-water.txt')
h28 = np.loadtxt('28per-cent_heavy-water.txt')
h58 = np.loadtxt('58per-cent_heavy-water.txt')

Q1 = water[:, 0]
I1 = water[:, 1]
Q2 = hwater[:, 0]
I2 = hwater[:, 1]
Q3 = h28[:, 0]
I3 = h28[:, 1]
Q4 = h58[:, 0]
I4 = h58[:, 1]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('ln(I) vs. $Q^2$')

ax1.plot(Q1*Q1, np.log(I1), '--o')
ax1.set_title('Pure water')
ax1.set_xlabel('$Q^2$')
ax1.set_ylabel('$ln(I)$')
ax1.set_xlim(0, 0.00004)
#ax1.set_ylim(0, 7.6)

ax2.plot(Q2*Q2, np.log(I2), '--o')
ax2.set_title('Pure heavy water')
ax2.set_xlabel('$Q^2$')
ax2.set_ylabel('$ln(I)$')
ax2.set_xlim(0, 0.000025)

ax3.plot(Q3*Q3, np.log(I3), '--o')
ax3.set_title('28% heavy water')
ax3.set_xlabel('$Q^2$')
ax3.set_ylabel('$ln(I)$')
ax3.set_xlim(0, 0.00003)

ax4.plot(Q4*Q4, np.log(I4), '--o')
ax4.set_title('58% heavy water')
ax4.set_xlabel('$Q^2$')
ax4.set_ylabel('$ln(I)$')
ax4.set_xlim(0, 0.0000225)

plt.tight_layout()
plt.show()

# Pure water
Q1fit = np.extract(Q1*Q1 < 0.00004, Q1)
I1fit = np.extract(Q1*Q1 < 0.00004, I1)

slope1, intcpt1 = np.polyfit(Q1fit*Q1fit, np.log(I1fit), deg=1)

plt.plot(Q1fit*Q1fit, np.log(I1fit), '--o')
plt.xlim(0, 0.00004)
plt.ylim(0, 7.6)

x1fit = np.linspace(0, 0.00004, 100)
y1fit = intcpt1 + x1fit * slope1

plt.plot(Q1fit*Q1fit, np.log(I1fit), 'o')
plt.plot(x1fit, y1fit)

# Pure heavy water
Q2fit = np.extract(Q2*Q2 < 0.000025, Q2)
I2fit = np.extract(Q2*Q2 < 0.000025, I2)

slope2, intcpt2 = np.polyfit(Q2fit*Q2fit, np.log(I2fit), deg=1)
x2fit = np.linspace(0, 0.000025, 100)
y2fit = intcpt2 + x2fit * slope2

plt.plot(Q2fit*Q2fit, np.log(I2fit), 'o')
plt.plot(x2fit, y2fit)

# 28% heavy water
Q3fit = np.extract(Q3*Q3 < 0.00003, Q3)
I3fit = np.extract(Q3*Q3 < 0.00003, I3)

slope3, intcpt3 = np.polyfit(Q3fit*Q3fit, np.log(I3fit), deg=1)
x3fit = np.linspace(0, 0.00003, 100)
y3fit = intcpt3 + x3fit * slope3

plt.plot(Q3fit*Q3fit, np.log(I3fit), 'o')
plt.plot(x3fit, y3fit)

# 58% heavy water
Q4fit = np.extract(Q3*Q3 < 0.0000225, Q4)
I4fit = np.extract(Q3*Q3 < 0.0000225, I4)

slope4, intcpt4 = np.polyfit(Q4fit*Q4fit, np.log(I4fit), deg=1)
x4fit = np.linspace(0, 0.0000225, 100)
y4fit = intcpt4 + x4fit * slope4

plt.plot(Q4fit*Q4fit, np.log(I4fit), 'o')
plt.plot(x4fit, y4fit)

#-------------------------------------------------
# Fits
fig, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2)
fig.suptitle('Fitted data')

ax5.plot(Q1fit*Q1fit, np.log(I1fit), 'o')
ax5.plot(x1fit, y1fit)
ax5.set_title('Pure water')
ax5.set_xlabel('$Q^2$')
ax5.set_ylabel('$ln(I)$')
ax5.set_xlim(0, 0.00004)

ax6.plot(Q2fit*Q2fit, np.log(I2fit), 'o')
ax6.plot(x2fit, y2fit)
ax6.set_title('Pure heavy water')
ax6.set_xlabel('$Q^2$')
ax6.set_ylabel('$ln(I)$')
ax6.set_xlim(0, 0.000025)

ax7.plot(Q3fit*Q3fit, np.log(I3fit), 'o')
ax7.plot(x3fit, y3fit)
ax7.set_title('28% heavy water')
ax7.set_xlabel('$Q^2$')
ax7.set_ylabel('$ln(I)$')
ax7.set_xlim(0, 0.00003)

ax8.plot(Q4fit*Q4fit, np.log(I4fit), 'o')
ax8.plot(x4fit, y4fit)
ax8.set_title('58% heavy water')
ax8.set_xlabel('$Q^2$')
ax8.set_ylabel('$ln(I)$')
ax8.set_xlim(0, 0.0000225)

plt.tight_layout()
plt.show()
#---------------------------------------------

plt.title('Intensity fits')
plt.xlabel('q')
plt.ylabel('Intensity')
plt.plot(Q1fit*Q1fit, np.log(I1fit), 'o', label='Pure water')
plt.plot(x1fit, y1fit)
plt.plot(Q2fit*Q2fit, np.log(I2fit), 'o', label='Pure heavy water')
plt.plot(x2fit, y2fit)
plt.plot(Q3fit*Q3fit, np.log(I3fit), 'o', label='28% heavy water')
plt.plot(x3fit, y3fit)
plt.plot(Q4fit*Q4fit, np.log(I4fit), 'og', label='58% heavy water')
plt.plot(x4fit, y4fit)
plt.legend()
plt.tight_layout()
plt.show()

Iplot = np.array([np.sqrt(np.exp(intcpt1)), np.sqrt(np.exp(intcpt3)),
                  np.sqrt(np.exp(intcpt4)), np.sqrt(np.exp(intcpt2))])
w = np.array([0, 28, 58, 100])

Iplot[3] = -Iplot[3]

plt.plot(w, Iplot, '--og')
plt.tight_layout()
plt.show()

m, n = np.polyfit(w, Iplot, deg=1)
xplot = np.linspace(0, 100, 100)
yplot = n + m * xplot

plt.title('Intensity fit for q=0')
plt.xlabel('Heavy water / %')
plt.ylabel('I(q=0)')
plt.plot(w, Iplot, 'o')
plt.plot(xplot, yplot)
plt.tight_layout()
plt.show()

contrast = -n/m

Mnipam = 120
dnipam = 1.1e6
Na = 6.022e23
H = -3.7406e-15
D = 6.671e-15
C = 6.6511e-15
N = 9.37e-15
O = 5.803e-15

Mh2o = 18
Md2o = 20

dh2o = 1e6
dd2o = 1.11e6

rhonipam = (6*C+7*D+4*H+N+O)/(Mnipam/(dnipam*Na))*(1e-20)
rhosolvent = ((contrast/100) * (2*D+O)/(Md2o/(dd2o*Na)) + (1-contrast/100) * (2*H+O)/(Mh2o/(dd2o*Na))) * 1e-20


# Function f as a piecewise function
def f(r, C, R):
    if 0 <= r <= R:
        return C
    else:
        return 0


# function g as a Gaussian
def g(r, s):
    return np.exp(-r**2 / (2*s**2))


# Convolution function
def f_convo(f, g, C, R, s, t):
    int = 0
    a = -np.inf
    b = np.inf
    N = 1000
    r_values = np.linspace(-5*s, 5*s, N)
    dr = r_values[1] - r_values[0]
    for r in r_values:
        int += f(r, C, R) * g(t - r, s) * dr
    return int


# Param
C = 1
R = 3
s = 0.5

# Defining range of t
tval = np.linspace(0, 2*R, 1000)

# Compute convolution value for each t
convoval = []
for t in tval:
    convolution = f_convo(f, g, C, R, s, t)
    convoval.append(convolution)

# Plot the results
plt.plot(tval, convoval)
plt.xlabel('r')
plt.ylabel('f(r)')
plt.title('Profile of a fuzzy sphere')
plt.tight_layout()
plt.show()
