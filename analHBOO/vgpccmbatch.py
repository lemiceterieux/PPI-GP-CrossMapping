import numpy as np
import numba as nb
import dcor
import scipy.stats as stats
import scipy.special as sp
import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)

class GP():
    def __init__(self):
        self.parameters = None
        self.muMat = None
#        self.muMat = "fix" 
        self.inpk = None
        self.cuda = 0
        self.noise = 1
        self.sigma = 4
        self.ld = 1
        self.A = None
        return

    def setcuda(self, cuda):
        self.cuda = cuda
   
    def testStateSpaceCorrelation(self, X, Y, Mu, m=3, tau=1, cuda=0, inc=.7):
        Z = []
        if not X is torch.Tensor:
            X = torch.from_numpy(X).float().cuda(cuda)
            Mu = torch.from_numpy(Mu).float().cuda(cuda)
            for z in Y:
                Z += [torch.from_numpy(np.array(z)).float().cpu()]
        x = (X - X.mean())/X.std()
        mu = (Mu - Mu.mean())/Mu.std()
        y = []
        for z in Z:
            y += [((z - z.mean())/z.std())]
   
        # State space transform
        xar = torch.stack([x[:,tau*i:tau*i-tau*m] for i in range(m)],1).cuda(cuda)
        xar = xar.reshape(m,-1).T#-1,xar.shape[-1]).T
        muar = torch.stack([mu[:,tau*i:tau*i-tau*m] for i in range(m)],1).cuda(cuda)
        muar = muar.reshape(m,-1).T#-1,muar.shape[-1]).T
        yar = []
        yp = []
        for z in y:
            yar += [torch.stack([z[:,tau*i:tau*i-tau*m] for i in range(m)],1).cpu()]
            yar[-1] = yar[-1].reshape(m,-1).T#-1,yar[-1].shape[-1]).T
            yar[-1] = yar[-1][:-1]
        yar = torch.stack(yar).cuda(cuda)

        # Train outs
        xp = xar[1:]#.matmul(simMats[0])
        xi = xar[:-1]#.matmul(simMats[0])
        mup = muar[1:]#.matmul(simMats[0])
        mui = muar[:-1]#.matmul(simMats[0])


        # Bayesian Regression
        mx = []
        kx = []
    
        gpx = self
        gpx.setcuda(cuda)
        li = []
        lt = []
        expY = []
    
        ml = []
        kl = []
        eeY = []
        W = torch.linalg.inv(yar.transpose(-1,-2).matmul(yar)).matmul(yar.transpose(-1,-2)).matmul(xi)
        m, k= gpx.forward(xi, xp, yar.matmul(W),mup)#.matmul(s))
        mx = m.transpose(-1,-2)
        if len(k.shape) > 1:
            kx = k.transpose(-1,-2)
        else:
            kx = k
        corr = []
        conditionals = []
        return mx, kx.squeeze()

    # Kernels
    def maternKernel(self, a, b, ld=3):
        r = torch.sqrt(batch_distance(a[None,:,:],b[None,:,:], ld))
        cmat = (1 + np.sqrt(5) * r + 5*r**2/(3))
        cmat = cmat * ( torch.exp(-np.sqrt(5)*r))
        #cmat[cmat!=cmat] = 0
        return cmat.T.squeeze()

    def squaredExpKernel(self, a, b, ld=3):
        r = batch_distance(a[None,:,:],b[None,:,:], ld)
        cmat = torch.exp(-r)
        return cmat.T.squeeze()

    def squaredExpKernelARD(self, a, b, ld=3):
        r = 0
        for i in range(len(ld)):
            temp = torch.cdist(a[...,[i]], b[...,[i]])#batch_distance(a[None,:,[i]],b[None,:,[i]],ld[i])
            r += temp**2/ld[i]#**2
        #temp = torch.cdist(a[None,:,:], b[None,:,:])
        #r = temp**2/ld[0]
        cmat = torch.exp(-r)
        return cmat.transpose(-1,-2).squeeze()

    def optimizeHyperparms(self, data, inp,mu, lr=.001, ld=1., sigma=6., noise=.1, niter=20, ns=False, kernel=None, m=40):
        data.requires_grad_(False)
        inp.requires_grad_(False)
        mu.requires_grad_(False)
        if kernel is None:
            kernel = self.maternKernel
        data = (data)
        ip = inp
        train = data

        A = (torch.randn(inp.shape[-1],inp.shape[-1])).float().cuda(self.cuda).requires_grad_()
        V = torch.linalg.svd(A)[-1]
        sigma = torch.tensor(sigma).float().log().cuda(self.cuda).requires_grad_()
        noiseo = torch.tensor(noise).float().cuda(self.cuda).requires_grad_()
        hnoise = (noise*(torch.ones(m).float())).cuda(self.cuda).requires_grad_()
        noise = torch.tensor(1*noise).float().cuda(self.cuda).requires_grad_()
        sld = (1*torch.ones(inp.shape[-1])).float().cuda(self.cuda).requires_grad_()
#        eld = (1*ld*torch.ones(inp.shape[-1])).float().cuda(self.cuda).requires_grad_()
        ld = (ld*torch.ones(inp.shape[-1])).float().log().cuda(self.cuda).requires_grad_()

        ssigma = (1*torch.ones(1)).float().cuda(self.cuda).requires_grad_()

        p = np.random.permutation(len(data))[:m]

        wa = 1
        eye = torch.eye(len(inp)).cuda(self.cuda).float()
        inp = ip.matmul(V)
        sipoints = (1*torch.ones(*inp[p].shape)).cuda(self.cuda).requires_grad_()
        ipoints = torch.randn_like(inp[p].detach()).cuda(self.cuda).requires_grad_()

        msigma = sigma.data.requires_grad_(False)
        mipoints = ipoints.data.requires_grad_(False)
        mld = ld.data.requires_grad_(False)
        parms = [sigma,ssigma,ld,sld,ipoints,sipoints]
        
        mSamples =1
        for i in range(niter):
            ls = []
            for k in range(mSamples):
                t = 1#0 if k == 0 else 1
                V = torch.linalg.svd(A)[-1]
                inp = ip.matmul(V)
                ripoints = (ipoints+(t*sipoints*torch.randn(*ipoints.shape).cuda(self.cuda)))
                rld = (ld+(t*sld*torch.randn(*ld.shape).cuda(self.cuda))).exp()
                rhnoise = hnoise
                noiseo = noiseo
                rsigma = (sigma+(t*ssigma*torch.randn(1).cuda(self.cuda))).exp()
                Km = rsigma*kernel(ripoints, ripoints, rld) + (hnoise**2).diag()
                Kn = rsigma*kernel(inp, inp, rld)
                Kmi = torch.inverse(Km)
                Knm = rsigma*kernel(inp, ripoints, rld)
                lamb =  (Kn - Knm.T.matmul(Kmi).matmul(Knm)).diag().diag()
                trr = train.matmul(V)
                mrr = mu.matmul(V)
                sqtrmr = (noise**2)*(1-torch.exp(-(1/ld.exp().detach()*(trr-mrr)**2)).mean(1))
                K = Knm.T.matmul(Kmi).matmul(Knm)  + lamb + (sqtrmr+noiseo**2)*eye
                Kinv = torch.inverse(K)
    
                logp = 0
                for tr,mm in zip(trr.T,mrr.T):
                    logp += -.5*(tr).T.matmul(Kinv).matmul(tr)
#                logp += -.5*(sqtrmr).T.matmul(Kinv).matmul(sqtrmr)
                logp = (logp/train.shape[-1] - torch.linalg.cholesky(K).slogdet()[1]).mean()
                (logp/mSamples).backward()
                ls += [logp.item()]
            logp =  -1/2*((-msigma + sigma)**2 -1 + ssigma**2) - torch.log(ssigma/1)#-(torch.log(ssigma/1) + (1 + sigma**2)/(2*ssigma**2) - 1/2)
            logp -= (1/2*((-mld + ld)**2 -1 + sld**2) - torch.log(sld/1)).sum()# (torch.log(sld/1) + (1 + ld**2)/(2*sld**2) - 1/2).mean()
            logp -= (1/2*((-mipoints + ipoints)**2 -1 + sipoints**2) - torch.log(sipoints/1)).sum()# (torch.log(sipoints/1) + (1 + ipoints**2)/(2*sipoints**2) - 1/2).mean()

            logp.backward()
            torch.nn.utils.clip_grad_norm_(parms,1000)
#            print(np.mean(ls),np.std(ls))
#            #print(ld,sld,sigma,ssigma, sigma.grad, ssigma.grad)
#            print(hnoise.grad.norm(),noiseo.grad.item(), ld.grad.norm(), ipoints.grad.norm())
#            print(sld.grad.norm(), ssigma.grad, sipoints.grad.norm())


            with torch.no_grad():
                sigma += lr*sigma.grad
                ld += lr*ld.grad
#                eld += lr*eld.grad
                noiseo += lr*noiseo.grad
                noise += lr*noise.grad
                hnoise += 10000*lr*hnoise.grad
                ipoints += lr*ipoints.grad
                ssigma += lr*ssigma.grad
                sld += lr*sld.grad
                sipoints += lr*sipoints.grad
                A += lr*A.grad



            sigma.grad.zero_()
            ld.grad.zero_()
#            eld.grad.zero_()
            noiseo.grad.zero_()
            noise.grad.zero_()
            hnoise.grad.zero_()
            ipoints.grad.zero_()
            ssigma.grad.zero_()
            sld.grad.zero_()
            sipoints.grad.zero_()
            A.grad.zero_()

        V = torch.linalg.svd(A)[-1]
        trr = train.matmul(V)
        mrr = mu.matmul(V)
        sqtrmr = (noise**2)*(1-torch.exp(-(1/ld.exp().detach()*(trr-mrr)**2).mean(1)))

        inp = ip.matmul(V)
        ripoints = (ipoints+(sipoints*torch.randn(*ipoints.shape).cuda(self.cuda)))
        rld = (ld+(sld*torch.randn(*ld.shape).cuda(self.cuda))).exp()
        rsigma = (sigma+(ssigma*torch.randn(1).cuda(self.cuda))).exp()

        Km = rsigma*kernel(ripoints, ripoints, rld) + (hnoise**2).diag()

        Kn = rsigma*kernel(inp, inp, rld)
        Kmi = torch.inverse(Km)
        Knm = rsigma*kernel(inp, ripoints, rld)
        lamb =  (Kn - Knm.T.matmul(Kmi).matmul(Knm)).diag().diag()
        K = Knm.T.matmul(Kmi).matmul(Knm)  + lamb +  (sqtrmr + noiseo**2)*eye
        Kinv = torch.inverse(K)
        
        ld = ld.detach().requires_grad_(False)
        self.sld = sld.detach().requires_grad_(False)

        sigma = sigma.detach().requires_grad_(False)
        self.ssigma = ssigma.detach().requires_grad_(False)

        noiseo = (sqtrmr +noiseo**2).detach().requires_grad_(False)
        noise = noise.detach().requires_grad_(False)**2
        self.A = V.detach().requires_grad_(False)
        self.hnoise = hnoise.detach()**2
        ipoints = ipoints.detach().requires_grad_(False)
        self.sipoints = sipoints.detach().requires_grad_(False)

        obsparms = [ld, sigma.item(),  noiseo, noise.item(), ipoints]

        logp = 0
        for tr in trr.T:
            logp += -.5*tr.T.matmul(Kinv).matmul(tr)
        logp = (logp/train.shape[-1] - torch.linalg.cholesky(K).slogdet()[1]).mean()
        self.logp = logp.item()
        Qm = Km + Knm.matmul(torch.inverse(lamb + noiseo*eye)).matmul(Knm.T)
        muMatrix = torch.inverse(Qm).matmul(Knm).matmul(torch.inverse(lamb + noiseo*eye)).matmul(train)
        stdMatrix = torch.inverse(Km) - torch.inverse(Qm)
        return logp, obsparms, muMatrix.detach(), stdMatrix.detach()

    # Posterior Inference
    def forward(self, inp, observations, test,mu, ld=1, sig=10, noise=.1, target="squaredexp"):
        # Choose kernel
        if target == "squaredexp":
            kernel = self.squaredExpKernelARD
        elif target == "ns_squaredexp":
            kernel = lambda t, i, l: self.squaredExpKernel(t,i,l) + self.WienerKernel(t,i,l)
        elif target == "matern":
            kernel = self.maternKernel
        elif target == "schoenberg":
            kernel = self.schoenbergKernel
        elif target == "ns_matern":
            kernel = lambda t, i, l: self.maternKernel(t,i,l) + self.WienerKernel(t,i,l)
        else:
            kernel = self.maternKernel

        ltest = test
        # Get Hyperparms
        if self.muMat is None:
            lp = []
            nsl = []
            noise, sigma, ld = [1,inp.std().item()**(1/2), (inp.std().item())**(1/2)]
            lp, parms, muMatrix, stdMatrix = self.optimizeHyperparms(observations, inp,mu, sigma=sigma, noise=noise, ld=ld ,niter=80,lr=1e-4, kernel=kernel)
            self.sigma = parms[1]
            self.ld = parms[0]
            self.noise = parms[2]
            self.ipoints = parms[-1]
            self.parms = parms
            self.stdMat = stdMatrix
            self.muMat = muMatrix

        posteriorK =[] 
        ltest = ltest.matmul(self.A)
        # Test kernel
            
        for i in range(100):
            if i == 0:
                t = 0
            else:
                t = 1
                p = np.random.permutation(len(ltest.ravel()))
                ltest = ltest.reshape(-1)[p].reshape(*ltest.shape).cuda(self.cuda)
            sigma = (self.sigma+(self.ssigma*np.random.randn()*t)).exp()
#            print(self.ld.shape, ltest.shape)
#            import time
#            time.sleep(100)
            ld = (self.ld +(self.sld*torch.randn(*self.ld.shape).cuda(self.cuda)*t)).exp()
            ipoints = self.ipoints+(self.sipoints*torch.randn(*self.ipoints.shape).cuda(self.cuda)*t)
            tk = sigma*kernel(ltest, ltest, ld)

            # Test given train kernel
            ip = inp.matmul(self.A)
            inptk = sigma*kernel(ltest, ipoints, ld)
            Km = sigma*kernel(ipoints, ipoints, ld) + (self.hnoise).diag()
            Knm = sigma*kernel(ip,ipoints,ld)
            Kmi = torch.inverse(Km) 
            Kn = sigma*kernel(ip,ip,ld)
            lamb =  (Kn - Knm.T.matmul(Kmi).matmul(Knm)).diag().diag()
            eye = torch.eye(tk.shape[-1]).cuda(self.cuda)
            lni = (1/(lamb+self.noise*eye).diag()).diag()
            Qm = Km + Knm.matmul(lni).matmul(Knm.T)
            stdMat= Kmi - torch.inverse(Qm)
           
            # Covariance update
            posteriorK += [(tk - inptk.transpose(-2,-1).matmul(stdMat).matmul(inptk) + self.noise*eye).slogdet()[1].cpu()]
        posteriorK = torch.stack(posteriorK)
        # Predicted test vals
        posteriormu = inptk.transpose(-2,-1).matmul(self.muMat)


        return posteriormu, posteriorK
