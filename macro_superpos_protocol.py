import numpy as np
import matplotlib.pyplot as plt


class Protocol():
    def __init__(self, lambda_f, omega_f, temp, Q, T2, Beta, n_imp, nc):
        self.lambda_w = 2 * np.pi * lambda_f
        self.omega_w = 2 * np.pi * omega_f
        self.temp = temp
        self.q = Q
        self.t2 = T2
        self.b = Beta 
        self.n_imp = n_imp
        self.nc = nc
        self.epsilon = 0
        self.n_th = ((1.381E-23 * self.temp)/(1.055E-34 * self.omega_w))

        self.delta = np.sqrt(0.5 * (self.n_imp + 0.5))
        self.tau = 2*np.pi * self.nc / self.omega_w
        self.gamma = self.omega_w / (2*np.pi * self.q)
        self.xi = np.sqrt(self.gamma**2 - 4*self.lambda_w**2 + 8*1j*self.gamma*self.lambda_w*(self.n_th + 0.5))
        self.hplus = (-self.gamma + self.xi)/(2*self.gamma * (self.n_th + 0.5) + 1j*self.lambda_w)
        self.hmin = (-self.gamma - self.xi)/(2*self.gamma * (self.n_th + 0.5) + 1j*self.lambda_w)

    def dif(self, _t, _g0):
        return _g0 * (self.n_th + 0.5) * (np.exp(-self.gamma*_t)-1) + np.exp(-self.gamma*_t) # inside A24

    def ddif(self, _t, _g0):
        return (_g0 * (np.exp(-self.xi*_t) - 1) + self.hplus - self.hmin*np.exp(-self.xi*_t)) / (self.hplus - self.hmin) # inside A25-A28

    def guu(self, _t, _g0):
        return _g0 / self.dif(_t,_g0) # A22

    def kuu(self, _t, _g0, _k0):
        return (_k0 * np.exp(-0.5*self.gamma*_t + 1j*(self.omega_w + self.lambda_w)*_t)) / (self.dif(_t,_g0)) # A23

    def quu(self, _t, _g0, _q0):
        return (_q0 * np.exp(-0.5*self.gamma*_t - 1j*(self.omega_w + self.lambda_w)*_t)) / (self.dif(_t,_g0)) # A23, but with q and complex conjugate of kuu

    def cuu(self, _t, _g0, _q0, _k0, _c0):
        return _c0 - np.emath.log(self.dif(_t,_g0)) + (_q0*_k0 * (self.n_th + 0.5) * (1-np.exp(-self.gamma*_t))) / (self.dif(_t,_g0)) # A24

    def kdd(self, _t, _g0, _k0):
        return (_k0 * np.exp(-0.5*self.gamma*_t + 1j*(self.omega_w - self.lambda_w)*_t)) / (self.dif(_t,_g0)) # A23

    def qdd(self, _t, _g0, _q0):
        return (_q0 * np.exp(-0.5*self.gamma*_t - 1j*(self.omega_w - self.lambda_w)*_t)) / (self.dif(_t,_g0)) # A23, but with q and complex conjugate of kdd

    def gud(self, _t, _g0):
        return (_g0*(self.hplus*np.exp(-self.xi*_t)-self.hmin) + self.hplus*self.hmin*(1-np.exp(-self.xi*_t))) / ((self.hplus-self.hmin)*self.ddif(_t,_g0)) # A25

    def kud(self, _t, _g0, _k0):
        return (_k0*np.exp(-0.5*self.xi*_t + 1j*self.omega_w*_t)) / (self.ddif(_t,_g0)) # A27

    def qud(self, _t, _g0, _q0):
        return (_q0*np.exp(-0.5*self.xi*_t - 1j*self.omega_w*_t)) / (self.ddif(_t,_g0)) # A26

    def cud(self, _t, _g0, _q0, _k0, _c0):
        return _c0 + np.emath.log(self.ddif(_t,_g0)) + (_q0*_k0*(1-np.exp(-self.xi*_t)))/((self.hplus-self.hmin)*self.ddif(_t,_g0)) + (1j*self.lambda_w+0.5*(self.gamma-self.xi)-1/self.t2 -1j*self.epsilon)*_t # A28, DEZE - VOOR DE LOG KLOPT NIET? en epsilon is niet meegenomen

    def wuu(self, _x, _p, _t, _g0, _k0, _q0, _c0):
        return np.exp(self.guu(_t,_g0)*(_x**2+_p**2) + self.kuu(_t,_g0,_k0)*(_x+1j*_p) + self.quu(_t,_g0,_q0)*(_x-1j*_p) + self.cuu(_t,_g0,_q0,_k0,_c0)) # A 21

    def wdd(self, _x, _p, _t, _g0, _k0, _q0, _c0):
        return np.exp(self.guu(_t,_g0)*(_x**2+_p**2) + self.kdd(_t,_g0,_k0)*(_x+1j*_p) + self.qdd(_t,_g0,_q0)*(_x-1j*_p) + self.cuu(_t,_g0,_q0,_k0,_c0)) # A 21, gdd=guu, cuu=cdd

    def wud(self, _x, _p, _t, _g0, _k0, _q0, _c0):
        return np.exp(self.gud(_t,_g0)*(_x**2+_p**2) + self.kud(_t,_g0,_k0)*(_x+1j*_p) + self.qud(_t,_g0,_q0)*(_x-1j*_p) + self.cud(_t,_g0,_q0,_k0,_c0)) # A 21

    def ppuu(self, _x, _t, _g0, _k0, _q0, _c0):
        return np.emath.sqrt((np.pi)/(-self.guu(_t,_g0)))*1/(self.dif(_t,_g0))*np.exp(self.guu(_t,_g0)*(_x + (_k0*np.exp(1j*(self.omega_w+self.lambda_w)*_t) + _q0*np.exp(-1j*(self.omega_w+self.lambda_w)*_t)*np.exp(-0.5*self.gamma*_t))/(2*_g0))**2 + _c0 - (_k0*_q0)/(_g0)) # A ??

    def ppdd(self, _x, _t, _g0, _k0, _q0, _c0):
        return np.emath.sqrt((np.pi)/(-self.guu(_t,_g0)))*1/(self.dif(_t,_g0))*np.exp(self.guu(_t,_g0)*(_x + (_k0*np.exp(1j*(self.omega_w-self.lambda_w)*_t) + _q0*np.exp(-1j*(self.omega_w-self.lambda_w)*_t)*np.exp(-0.5*self.gamma*_t))/(2*_g0))**2 + _c0 - (_k0*_q0)/(_g0)) # A ??

    def factors(self):
        self.GUU = self.guu(self.tau, -1/(2*self.delta**2))
        self.KUU = self.kuu(self.tau, -1/(2*self.delta**2), -1j*self.b / (2*self.delta**2))
        self.QUU = self.quu(self.tau, -1/(2*self.delta**2), 1j*self.b / (2*self.delta**2))
        self.CUU = self.cuu(self.tau, -1/(2*self.delta**2), 1j*self.b / (2*self.delta**2), -1j*self.b / (2*self.delta**2), -self.b**2/(2*self.delta**2) - np.log(4*np.pi*self.delta**2))
        self.KDD = self.kdd(self.tau, -1/(2*self.delta**2), -1j*self.b / (2*self.delta**2))
        self.QDD = self.qdd(self.tau, -1/(2*self.delta**2), 1j*self.b / (2*self.delta**2))
        self.GUD = self.gud(self.tau, -1/(2*self.delta**2))
        self.KUD = self.kud(self.tau, -1/(2*self.delta**2), -1j*self.b / (2*self.delta**2))
        self.QUD = self.qud(self.tau, -1/(2*self.delta**2), 1j*self.b / (2*self.delta**2))
        self.CUD = self.cud(self.tau, -1/(2*self.delta**2), 1j*self.b / (2*self.delta**2), -1j*self.b / (2*self.delta**2), -self.b**2/(2*self.delta**2) - np.log(-1j*4*np.pi*self.delta**2))

    def p2t(self, _x):
        self.factors()
        pu = 0.5*(self.ppuu(_x,self.tau,self.GUU,self.KUU,self.QUU,self.CUU) 
                + self.ppuu(_x,self.tau,self.GUU,self.KDD,self.QDD,self.CUU) 
             + 1j*self.ppuu(_x,self.tau,self.GUD,self.KUD,self.QUD,self.CUD)
             - 1j*self.ppuu(_x,self.tau,np.conj(self.GUD),np.conj(self.KUD),np.conj(self.QUD),np.conj(self.CUD)))
        pd = 0.5*(self.ppdd(_x,self.tau,self.GUU,self.KUU,self.QUU,self.CUU) 
                + self.ppdd(_x,self.tau,self.GUU,self.KDD,self.QDD,self.CUU) 
             - 1j*self.ppdd(_x,self.tau,self.GUD,self.KUD,self.QUD,self.CUD)
             + 1j*self.ppdd(_x,self.tau,np.conj(self.GUD),np.conj(self.KUD),np.conj(self.QUD),np.conj(self.CUD)))
        return pu + pd

    def p3t(self, _x):
        self.factors()
        GUUTUU = self.guu(self.tau,self.GUU)
        KUUTUU = self.kuu(self.tau,self.GUU,self.KUU)
        QUUTUU = self.quu(self.tau,self.GUU,self.QUU)
        CUUTUU = self.cuu(self.tau,self.GUU,self.QUU,self.KUU,self.CUU)

        KUUTDD = self.kuu(self.tau,self.GUU,self.KDD)
        QUUTDD = self.quu(self.tau,self.GUU,self.QDD)
        CUUTDD = self.cuu(self.tau,self.GUU,self.QDD,self.KDD,self.CUU)

        GUUTUD = self.guu(self.tau,self.GUD)
        KUUTUD = self.kuu(self.tau,self.GUD,self.KUD)
        QUUTUD = self.quu(self.tau,self.GUD,self.QUD)
        CUUTUD = self.cuu(self.tau,self.GUD,self.QUD,self.KUD,self.CUD)

        GUUTDU = self.guu(self.tau,np.conj(self.GUD))
        KUUTDU = self.kuu(self.tau,np.conj(self.GUD),np.conj(self.KUD))
        QUUTDU = self.quu(self.tau,np.conj(self.GUD),np.conj(self.QUD))
        CUUTDU = self.cuu(self.tau,np.conj(self.GUD),np.conj(self.QUD),np.conj(self.KUD),np.conj(self.CUD))

        KDDTUU = self.kdd(self.tau,self.GUU,self.KUU)
        QDDTUU = self.qdd(self.tau,self.GUU,self.QUU)
        KDDTDD = self.kdd(self.tau,self.GUU,self.KDD)
        QDDTDD = self.qdd(self.tau,self.GUU,self.QDD)

        KDDTUD = self.kdd(self.tau,self.GUD,self.KUD)
        QDDTUD = self.qdd(self.tau,self.GUD,self.QUD)

        KDDTDU = self.kdd(self.tau,np.conj(self.GUD),np.conj(self.KUD))
        QDDTDU = self.qdd(self.tau,np.conj(self.GUD),np.conj(self.QUD))

        GUDTUU = self.gud(self.tau,self.GUU)
        KUDTUU = self.kud(self.tau,self.GUU,self.KUU)
        QUDTUU = self.qud(self.tau,self.GUU,self.QUU)
        CUDTUU = self.cud(self.tau,self.GUU,self.QUU,self.KUU,self.CUU)

        KUDTDD = self.kud(self.tau,self.GUU,self.KDD)
        QUDTDD = self.qud(self.tau,self.GUU,self.QDD)
        CUDTDD = self.cud(self.tau,self.GUU,self.QDD,self.KDD,self.CUU)

        GUDTUD = self.gud(self.tau,self.GUD)
        KUDTUD = self.kud(self.tau,self.GUD,self.KUD)
        QUDTUD = self.qud(self.tau,self.GUD,self.QUD)
        CUDTUD = self.cud(self.tau,self.GUD,self.QUD,self.KUD,self.CUD)

        GUDTDU = self.gud(self.tau,np.conj(self.GUD))
        KUDTDU = self.kud(self.tau,np.conj(self.GUD),np.conj(self.KUD))
        QUDTDU = self.qud(self.tau,np.conj(self.GUD),np.conj(self.QUD))
        CUDTDU = self.cud(self.tau,np.conj(self.GUD),np.conj(self.QUD),np.conj(self.KUD),np.conj(self.CUD))


        pu = (0.25*(self.ppuu(_x,self.tau,GUUTUU,KUUTUU,QUUTUU,CUUTUU) 
                  + self.ppuu(_x,self.tau,GUUTUU,KUUTDD,QUUTDD,CUUTDD)
                +1j*self.ppuu(_x,self.tau,GUUTUD,KUUTUD,QUUTUD,CUUTUD)
                -1j*self.ppuu(_x,self.tau,GUUTDU,KUUTDU,QUUTDU,CUUTDU)
                  + self.ppuu(_x,self.tau,GUUTUU,KDDTUU,QDDTUU,CUUTUU)
                  + self.ppuu(_x,self.tau,GUUTUU,KDDTDD,QDDTDD,CUUTDD)
                -1j*self.ppuu(_x,self.tau,GUUTUD,KDDTUD,QDDTUD,CUUTUD)
                +1j*self.ppuu(_x,self.tau,GUUTDU,KDDTDU,QDDTDU,CUUTDU))
             - 0.5*(np.imag(1j*self.ppuu(_x,self.tau,GUDTUU,KUDTUU,QUDTUU,CUDTUU)
                           -1j*self.ppuu(_x,self.tau,GUDTUU,KUDTDD,QUDTDD,CUDTDD)
                            +  self.ppuu(_x,self.tau,GUDTUD,KUDTUD,QUDTUD,CUDTUD)
                            +  self.ppuu(_x,self.tau,GUDTDU,KUDTDU,QUDTDU,CUDTDU))))

        pd = (0.25*(self.ppdd(_x,self.tau,GUUTUU,KUUTUU,QUUTUU,CUUTUU) 
                  + self.ppdd(_x,self.tau,GUUTUU,KUUTDD,QUUTDD,CUUTDD)
                +1j*self.ppdd(_x,self.tau,GUUTUD,KUUTUD,QUUTUD,CUUTUD)
                -1j*self.ppdd(_x,self.tau,GUUTDU,KUUTDU,QUUTDU,CUUTDU)
                  + self.ppdd(_x,self.tau,GUUTUU,KDDTUU,QDDTUU,CUUTUU)
                  + self.ppdd(_x,self.tau,GUUTUU,KDDTDD,QDDTDD,CUUTDD)
                -1j*self.ppdd(_x,self.tau,GUUTUD,KDDTUD,QDDTUD,CUUTUD)
                +1j*self.ppdd(_x,self.tau,GUUTDU,KDDTDU,QDDTDU,CUUTDU))
             + 0.5*(np.imag(1j*self.ppdd(_x,self.tau,GUDTUU,KUDTUU,QUDTUU,CUDTUU)
                           -1j*self.ppdd(_x,self.tau,GUDTUU,KUDTDD,QUDTDD,CUDTDD)
                            +  self.ppdd(_x,self.tau,GUDTUD,KUDTUD,QUDTUD,CUDTUD)
                            +  self.ppdd(_x,self.tau,GUDTDU,KUDTDU,QUDTDU,CUDTDU))))

        p3t_re = np.real(pu + pd)
        max_left = np.max(p3t_re[:500])
        max_right = np.max(p3t_re[500:])
        self.fraction = max_left / max_right
        return pu + pd

    def calc_frac(self):
        x = np.linspace(-200, 200, 1001)
        self.p3t(x)
        return self.fraction

    def plot_p2t(self):
        x = np.linspace(-100, 100, 101)
        fig, ax = plt.subplots()
        ax.plot(x, self.p2t(x))
        ax.grid()
        plt.show()

    def plot_p3t(self):
        x = np.linspace(-200, 200, 1001)
        fig, ax = plt.subplots()
        ax.plot(x, np.real(self.p3t(x)))
        ax.grid()
        ax.set_ylabel("P(X)")
        ax.set_xlabel("X(n.u.)")
        ax.set_title(f"T = {self.temp*1000} mK, Q = {self.q:.2e}")
        plt.show()


def multiple_nc():
    x = np.linspace(-200, 200, 1001)
    y = np.zeros((len(x), 5))
    for i in range(5):
        simulation = Protocol(lambda_f = 0.045, 
                              omega_f = 3000, 
                              temp = 1E-3, 
                              Q = 1E8, 
                              T2 = 0.1, 
                              Beta = 1E5, 
                              n_imp = 760, 
                              nc = i+1)
        y[:,i] = np.real(simulation.p3t(x))
    
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.grid()
    ax.legend(["nc = 1", "nc = 2", "nc = 3", "nc = 4", "nc = 5"])
    ax.set_ylabel("P(X)")
    ax.set_xlabel("X(n.u.)")
    plt.show()


def multiple_nth():
    x = np.linspace(-200, 200, 1001)
    n_th_list = np.linspace(0,10,25) 
    temp_list = n_th_list * (1.055E-34 * (3000*2*np.pi)) / 1.381E-23
    fraction_list = np.zeros(25)
    for idx, t in enumerate(temp_list):
        simulation = Protocol(lambda_f = 0.045, 
                          omega_f = 3000, 
                          temp = t, 
                          Q = 1E5, 
                          T2 = 0.1, 
                          Beta = 1E5, 
                          n_imp = 760, 
                          nc = 5)
        fraction_list[idx] = simulation.calc_frac()

    fig, ax = plt.subplots()
    ax.scatter(n_th_list, fraction_list)
    ax.grid()
    ax.set_ylabel(r"$\eta$")
    ax.set_xlabel("n_th")
    plt.show()


def optimap_qt():
    def fracfunc(q, t):
        simulation = Protocol(lambda_f = 0.045, 
                          omega_f = 3000, 
                          temp = t, 
                          Q = q, 
                          T2 = 0.1, 
                          Beta = 1E5, 
                          n_imp = 760, 
                          nc = 5)
        return simulation.calc_frac()

    q_list = np.logspace(5,7,11)
    t_list = np.logspace(-4,-2,11)
    frac_im = np.zeros((11,11))
    for i, q in enumerate(q_list):
        for j, t in enumerate(t_list):
            frac_im[i,j] = fracfunc(q, t)
    fig, ax = plt.subplots()
    im = ax.imshow(frac_im, aspect="equal", origin="lower")
    plt.colorbar(im)
    ax.set_xticks(np.arange(11), ['{:.2f}'.format(float(x)) for x in 1000*t_list])
    ax.set_yticks(np.arange(11), ['{:.2e}'.format(float(x)) for x in q_list])
    ax.set_xlabel("temperature [mK]")
    ax.set_ylabel("Q-factor [-]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #multiple_nc()
    #multiple_nth()
    #optimap_qt()
    simulation = Protocol(lambda_f = 0.045, 
                          omega_f = 3000, 
                          temp = 2E-4, 
                          Q = 3E5, 
                          T2 = 0.1, 
                          Beta = 3.33E4, 
                          n_imp = 80, 
                          nc = 3)
    simulation.plot_p3t()
    print(simulation.fraction)