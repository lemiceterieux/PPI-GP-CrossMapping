import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import scipy.signal as signal
from joblib import Parallel, delayed
def simulate(kk):
    matplotlib.use('agg')
    SNR = 80
    Eo = .4
    alpha= .2
    tauv = 30
    tauo = 2
    fs = 100
    cutOffFreq = np.linspace(0.001,.7999,10)
#    traps = np.convolve(np.ones(int(10*fs)), np.ones(int(30*fs)))
    traps = np.convolve(np.ones(int(2*fs)), np.ones(int(5*fs)))
    win = signal.hamming(len(traps))
    traps = traps/traps.max()*1#*win
    u=np.zeros(int(fs*100*10))#.1*np.sin([i*2*np.pi*1/100 for i in range(int(fs*60*10))]) + np.random.rand(int(fs*60*10))*0.001
    print(len(u))
    #u = np.fft.rfft(u)
    #u = u*np.exp(-2*np.pi*np.random.rand(len(u))*1j)
    #u = np.fft.irfft(u)
    tauv = tauv + np.zeros(len(u))
    #u = np.convolve(np.random.binomial(5,.01,fs*60*10)*5, traps, 'same')
    #u = u/u.max()*10
    signs = [0,0]
    for i in range(20):
        if signs[0] >= 10 or signs[1] >= 10:
           s = np.argmin(signs)*2-1 
           print("YEAH",signs,s)
        else:
            s =  np.sign(np.random.randn())
            if s <0:
                signs[0] += 1
            else:
                signs[1] +=1
        u[int(fs*10)+i*fs*50:int(fs*10)+i*fs*50 + len(traps)] += s*traps
        u[int(2*fs*10)+i*fs*50:int(2*fs*10)+i*fs*50 + len(traps)] += -s*traps

    print(min(u),max(u))
    #u = np.clip(u + np.random.randn(len(u))*1, -50, 50)
    #u = u +  .5*np.sin([i*2*np.pi/10*.1 for i in range(len(u))])#.5*np.random.randn(len(u))
    #u[abs(u) < 1] = 0
    X = np.zeros((len(u),100))
    q = np.zeros((len(u),100))
    p = np.zeros((len(u),100))
    v = np.zeros((len(u),100))
    f = np.zeros((len(u),100))
    s = np.zeros((len(u),100))
    high = .1
    low = .01
    rHR = (high+low)*np.random.rand(len(u)) +low
    h = 1/fs
    np.random.seed(kk)
    A1 = .4*(np.random.randn(50,50))
    A2 = .4*(np.random.randn(50,50))
    A3 = .4*(np.random.randn(50,50))
    A4 = .4*(np.random.randn(50,50))
    A = np.eye(100)
    A[:50,:50] = A1
    A[50:,50:] = A3
    C = .1*np.random.randn(100,100)#np.eye(100)
    C2 = .1*np.random.randn(100,100)#np.eye(100)
    U2 = 10*np.random.binomial(1,.3,(len(u),100)).astype(float)
    U2[:,25:75] *= .1

    def dqdvdp(X,Q, V, P,S,F, U,t):
        Eo = .4
        tauo = 2
        tauv = 30
        epsilon = .005
        taus = .8
        scale = .4#+HRV[t]
        A = -np.eye(100)*2
        A[:50,:50] = A1
        A[50:,50:] = A3
        UU = U2[t]
        dx = -3*X +UU
        if U < 0:
            A[50:,:50] = 1*A2
            U = 1*np.ones(100)*U**2
            C[50:] *= 1
            U = (U)
            dx += (A.dot(X)) +U#C.dot(np.diag(U))
        elif U > 0:
            A[:50,50:] = 1*A4
            U = 1*np.ones(100)*U**2
            U = (U)
            C[:50] *= 1
            dx += (A.dot(X)) + U#C.dot(np.diag(U))
        U = abs(U)
        tauf = scale#.4+.39*scale
        E = 1 - (1 - Eo)**(1/F)
        alpha = .4
        F = np.log(1 + np.exp(F))

        df = S
        ds = epsilon*X - S/taus - (F-1)/tauf
        dq = F/tauo *(E/Eo - Q/V) + 1/tauv*(F - V**(1/alpha))*Q/V
        dv = 1/tauv*(F - V**(1/alpha))
        dp = 1/tauv*(F - V**(1/alpha))*P/V
        return np.vstack((dq, dv, dp, df, ds,dx))

    X[0] = 0*np.ones(100)#np.random.rand()*.1 
    q[0] = 1*np.ones(100)#np.random.rand()*.1 
    p[0] = 6.4*np.ones(100)#np.random.rand()*.1
    v[0] = 1*np.ones(100)#np.random.rand()*.1
    s[0] = 0*np.ones(100)#.1#np.random.rand()*.1
    f[0] = 1*np.ones(100)#.1#np.random.rand()*.1
    derivf = []
    derivs = []
    derivv = []
    derivp = []
    derivq = []
    derivx = []
    for i, (y1_t, y2_t, y3_t, y4_t, y5_t,y6_t) in enumerate(zip(q[:-1],v[:-1],p[:-1], s[:-1], f[:-1],X[:-1])):
    #    k1 = dqdvdp(q[i], v[i], p[i], s[i], f[i], u[i])
    #    derivq.append(k1[0])
    #    derivv.append(k1[1])
    #    derivp.append(k1[2])
    #    derivf.append(k1[3])
    #    derivs.append(k1[4])
    #    q[i+1] = y1_t + h*derivq[-1]# + np.random.randn()*dnoise
    #    v[i+1] = y2_t + h*derivv[-1]# + np.random.randn()*dnoise
    #    p[i+1] = y3_t + h*derivp[-1]# + np.random.randn()*dnoise
    #    s[i+1] = y4_t + h*derivs[-1]# + np.random.randn()*dnoise
    #    f[i+1] = y5_t + h*derivf[-1]# + np.random.randn()*dnoise
    
        if i < 4:
            k1 = dqdvdp(X[i],q[i], v[i], p[i], s[i], f[i], u[i],i)
            ink2 = (y6_t + k1[5]/2, y1_t + k1[0]/2, y2_t + k1[1]/2, y3_t + k1[2]/2,y4_t + k1[3]/2, y5_t + k1[4]/2,)
            k2 = dqdvdp(*ink2, u[i],i)
            ink3 = (y6_t + k2[5]/2, y1_t + k2[0]/2, y2_t + k2[1]/2, y3_t + k2[2]/2,y4_t + k2[3]/2, y5_t + k2[4]/2,)
            k3 = dqdvdp(*ink3, u[i],i)
            ink4 = (y6_t + k2[5], y1_t + k3[0], y2_t + k3[1], y3_t + k3[2], y4_t + k3[3], y5_t + k3[4],)
            k4 = dqdvdp(*ink4, u[i],i)
            derivq.append(1/6*(k1[0] + k2[0]*2 + 2*k3[0] + k4[0]))
            derivv.append(1/6*(k1[1] + k2[1]*2 + 2*k3[1] + k4[1]))
            derivp.append(1/6*(k1[2] + k2[2]*2 + 2*k3[2] + k4[2]))
            derivf.append(1/6*(k1[3] + k2[3]*2 + 2*k3[3] + k4[3]))
            derivs.append(1/6*(k1[4] + k2[4]*2 + 2*k3[4] + k4[4]))
            derivx.append(1/6*(k1[5] + k2[5]*2 + 2*k3[5] + k4[5]))
            q[i+1] = y1_t + h*derivq[-1]# + np.random.randn()*dnoise
            v[i+1] = y2_t + h*derivv[-1]# + np.random.randn()*dnoise
            p[i+1] = y3_t + h*derivp[-1]# + np.random.randn()*dnoise
            s[i+1] = y4_t + h*derivs[-1]# + np.random.randn()*dnoise
            f[i+1] = y5_t + h*derivf[-1]# + np.random.randn()*dnoise
            X[i+1] = y6_t + h*derivx[-1]# + np.random.randn()*dnoise
        else:
            pds = dqdvdp(X[i],q[i], v[i], p[i], s[i], f[i], u[i],i)
            predx = y1_t + h/24*(55*pds[5] - 59*derivx[-1] + 37*derivx[-2] - 9*derivx[-3])
            predq = y1_t + h/24*(55*pds[0] - 59*derivq[-1] + 37*derivq[-2] - 9*derivq[-3])
            predv = y2_t + h/24*(55*pds[1] - 59*derivv[-1] + 37*derivv[-2] - 9*derivv[-3])
            predp = y3_t + h/24*(55*pds[2] - 59*derivp[-1] + 37*derivp[-2] - 9*derivp[-3])
            predf = y5_t + h/24*(55*pds[3] - 59*derivf[-1] + 37*derivf[-2] - 9*derivf[-3])
            preds = y4_t + h/24*(55*pds[4] - 59*derivs[-1] + 37*derivs[-2] - 9*derivs[-3])
            cds = dqdvdp(predx,predq, predv, predp, preds, predf, u[i],i)
            derivx.append(1/24*(9*cds[5] + 19*pds[5] - 5*derivx[-1] + derivx[-2]))
            derivq.append(1/24*(9*cds[0] + 19*pds[0] - 5*derivq[-1] + derivq[-2]))
            derivv.append(1/24*(9*cds[1] + 19*pds[1] - 5*derivv[-1] + derivv[-2]))
            derivp.append(1/24*(9*cds[2] + 19*pds[2] - 5*derivp[-1] + derivp[-2]))
            derivf.append(1/24*(9*cds[3] + 19*pds[3] - 5*derivf[-1] + derivf[-2]))
            derivs.append(1/24*(9*cds[4] + 19*pds[4] - 5*derivs[-1] + derivs[-2]))
            X[i+1] = y6_t + h*derivx[-1]# + np.random.randn()*dnoise
            q[i+1] = y1_t + h*derivq[-1]# + np.random.randn()*dnoise
            v[i+1] = y2_t + h*derivv[-1]# + np.random.randn()*dnoise
            p[i+1] = y3_t + h*derivp[-1]# + np.random.randn()*dnoise
            s[i+1] = y4_t + h*derivs[-1]# + np.random.randn()*dnoise
            f[i+1] = y5_t + h*derivf[-1]# + np.random.randn()*dnoise
    
    #plt.plot(derivf)
    #plt.plot(s)
    #plt.plot(derivq)
    #plt.plot(derivp)
    #plt.plot(derivv)
    qpure = np.copy(q)
    ppure = np.copy(p)
    freqH = np.ones(len(u))*np.random.rand()*(3-.8)+.8
    for i in range(0,len(u),200):
        if np.mean(u[i:i+200].mean()) > .3:
            freqH[i:i+200] += np.random.randn()*.1
        else:
            freqH[i:i+200] += np.random.randn()*.01
    #freqH[~(u>0)] += np.random.randn(sum(~(u>0)))*.01
    #freqH[(u>0)] += np.random.randn(sum(u>0))*.1
    freqH1 = 2*freqH
    freqH2 = 2*freqH1
    pulse = np.sin((freqH)*2*np.pi*np.arange(len(u))/100)
    pulse += .3*np.sin((freqH1)*2*np.pi*np.arange(len(u))/100)
    pulse += .1*np.sin((freqH2)*2*np.pi*np.arange(len(u))/100)
    sampEns = [] 
    for SNR in [5]:
        qcons = (qpure-qpure.mean(0))/qpure.std(0)#+10**(-SNR/20)*np.random.randn(len(u),100)
        pcons = (ppure-ppure.mean(0))/ppure.std(0)#+10**(-SNR/20)*np.random.randn(len(u),100)
        b,a = signal.butter(3,.2/(fs/2), "low")
        q = signal.filtfilt(b,a,qcons,axis=0)
        p = signal.filtfilt(b,a,pcons,axis=0)
    np.save("control{0:d}.npy".format(kk),u)
    np.save("qpure{0:d}.npy".format(kk),qpure)
    np.save("xpure{0:d}.npy".format(kk),X)
    return q
sampEns = np.array(Parallel(n_jobs=100)(delayed(simulate)(k) for k in range(100)))
np.save("sampEns",sampEns)
#sampEns += np.random.randn(*sampEns.shape)*10**(-5/20)
control = np.load("control0.npy")
plt.subplot(3,1,1)
plt.plot(np.arange(sampEns.shape[1]-500)/100,sampEns[0,500:,:50])
plt.title("Population 1")
plt.subplot(3,1,2)
plt.plot(np.arange(sampEns.shape[1]-500)/100,sampEns[0,500:,50:])
plt.title("Population 2")
plt.subplot(3,1,3)
plt.plot(np.arange(sampEns.shape[1]-500)/100,control[500:])
plt.title("Stimuli")
plt.xlabel("Time (s)")
plt.savefig("lol.png")
