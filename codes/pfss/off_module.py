'''
Code       : off_module.py
Date       : 2024.11.01
Contributer: H.Y.Li (liyuhua0909@126.com), G.Y.Chen (gychen@smail.nju.edu.cn)
Purpose    : Build a Outflow Field magnetic field model...

### --------------------------------- ###
Remark:
2024.11.01: Build the code
2024.11.22: Modified the rc calculation
'''

from .needs import *
from scipy.special import lambertw

from .pfss_module import pfss_solver
from .funcs import brtp2bxyz, trilinear_interpolation, Brtp_lm, Associated_Legendre,DAssociated_Legendre
from .magline import rk45, magline_stepper, magline_solver, show_boundary, show_maglines,parallel_magline_solver,show_current_sheet

# ==========================================

L0  = 6.995e10
t0  = 5972.5794
v0  = 1.16448846777562e7
nu0 = 5.e-17*L0**2/t0
GM  = 9.54e4

def rk45_solver(rfun, x, y, dl, **kwargs):
    sig = kwargs.get('sig', 1)
    x0  = x
    k1  = rfun(x0         , y        , **kwargs)
    k2  = rfun(x0+sig*dl/2, y+sig*k1*dl/2, **kwargs)
    k3  = rfun(x0+sig*dl/2, y+sig*k2*dl/2, **kwargs)
    k4  = rfun(x0+sig*dl  , y+sig*k3*dl  , **kwargs)
    k   = (k1+2*k2+2*k3+k4)/6*sig
    ret = y+k*dl
    return ret

# ==========================================

class off_solver(pfss_solver):
    def __init__(self,
                 fits_file,
                 n_r      = 400,
                 n_t      = 200,
                 n_p      = 400,
                 lmax     = 80,
                 Rs       = 2.5,
                 v1       = 50
                ):
        super().__init__(fits_file,n_r,n_t,n_p,lmax,Rs)
        self.scs_file  = './Brtp_off.npy'
        self.save_name = 'off_solver.pkl'
        self.v1        = v1*1e5/v0
        self.rc        = self.cal_rc()
        self.off_file  = './Outflow_field.npy'

    def cal_rc(self):
        r1 = self.Rs
        v1 = self.v1*v0*1e-5
        r0 = GM/(2*v1**2)
        rc = np.where(r0>r1,
                      3/(r1/r0-4)*lambertw(1/3*(r1/r0-4)*(r1/r0)**(1/3)*np.exp(-1),k=-1)*r1,
                      3/(r1/r0-4)*lambertw(1/3*(r1/r0-4)*(r1/r0)**(1/3)*np.exp(-1),k= 0)*r1
                     )
        return np.real(rc)

    def D(self, r):
        rc = self.rc
        if isinstance(rc,np.ndarray):
            if rc.ndim==1:
                rc=rc[:,None]
        Dr = (rc/r)**4*np.exp(4*(1-rc/r)-1)
        return Dr

    def vout(self, rho):
        x = np.asarray(rho)-np.log(self.rc)
        sonic = np.abs(x) <= 2e-3
        # Evaluate the regular branch away from W=-1; use its local series below.
        safe_rho = np.where(sonic, np.log(self.rc)+.01, rho)
        r   = np.exp(safe_rho)
        rc  = self.rc
        cs  = np.sqrt(GM/2/self.rc)*1e5/v0
        if isinstance(cs,np.ndarray) and len(cs)>1:
            if cs.ndim==1:
                cs=cs[:,None]
                rc=rc[:,None]
        Dr  = self.D(r)
        ret = np.where(r<=rc, cs*np.real(np.sqrt(-lambertw(-Dr, k=0))), cs*np.real(np.sqrt(-lambertw(-Dr,k=-1))))
        return np.where(sonic, cs*(1+x-x**3/12+x**4/20-3*x**5/160+19*x**6/6720), ret)

    def d2_vout(self, rho):
        # Differentiate the transonic wind equation with respect to log(r).
        x = np.asarray(rho)-np.log(self.rc)
        sonic = np.abs(x) <= 2e-3
        safe_rho = np.where(sonic, np.log(self.rc)+.01, rho)
        r = np.exp(safe_rho)
        v = self.vout(safe_rho)
        dv = self.d1_vout(safe_rho)
        cs = np.sqrt(GM/2/self.rc)*1e5/v0
        u = (v/cs)**2
        a = dv/v
        da = 2*self.rc/r/(u-1)-4*(1-self.rc/r)*u*a/(u-1)**2
        return np.where(sonic, cs*(-x/2+3*x*x/5-3*x**3/8+19*x**4/224), v*(a*a+da))

    def d1_vout(self, rho):
        x = np.asarray(rho)-np.log(self.rc)
        sonic = np.abs(x) <= 2e-3
        # Evaluate the regular branch away from W=-1; use its local series below.
        safe_rho = np.where(sonic, np.log(self.rc)+.01, rho)
        r   = np.exp(safe_rho)
        rc  = self.rc
        Dr  = self.D(r)
        w   = np.where(r<=rc, np.real(lambertw(-Dr, k=0)), np.real(lambertw(-Dr, k=-1)))
        cs  = np.sqrt(GM/2/self.rc)*1e5/v0
        dDr = -4*(1-rc/r)*Dr
        # Derivative with respect to rho=log(r), on either Lambert-W branch.
        ret = cs*0.5*(-w)**0.5/Dr/(1+w)*dDr
        return np.where(sonic, cs*(1-x*x/4+x**3/5-3*x**4/32+19*x**5/1120), ret)

    # def rfun(self, rho, y=None, l=0, **kwargs):
    #     y0,y1 = y
    #     k1 = (3-nu0*np.exp(rho)*self.vout(rho))
    #     k2 = -(l*(l+1)-2+3*nu0*np.exp(rho)*self.vout(rho)+nu0*np.exp(rho)*self.d2_vout(rho))
    #     rf1 = -k1*y1-k2*y0
    #     rf0 = y1
    #     ret = np.array([rf0,rf1])
    #     return ret

    # def rfun(self, rho, y=None, l=0, **kwargs):
    #     y0,y1 = y
    #     k1  = (4-(2-nu0)*np.exp(rho)*self.vout(rho))
    #     k2  = -(l*(l+1)-3+(5-nu0)*np.exp(rho)*self.vout(rho)-(1-nu0)*np.exp(2*rho)*self.vout(rho)**2+np.exp(rho)*self.d1_vout(rho))
    #     rf1 = -k1*y1-k2*y0
    #     rf0 = y1
    #     ret = np.array([rf0,rf1])
    #     return ret

    def rfun(self, rho, y=None, l=0, **kwargs):
        y0,y1 = y
        k1 = (3-nu0*np.exp(rho)*self.vout(rho))
        k2 = -(l*(l+1)-2+3*nu0*np.exp(rho)*self.vout(rho)+nu0*np.exp(rho)*self.d1_vout(rho))
        rf1 = -k1*y1-k2*y0
        rf0 = y1
        ret = np.array([rf0,rf1])
        return ret

    def time_integration(self, initial, **kwargs):
        r1 = kwargs.get('r1', self.Rs)
        dl = kwargs.get('dl', 1e-3)
        l = kwargs.get('l', 0)
        ns = kwargs.get('max_steps', 100000)
        rmax = kwargs.get('rmax', r1)
        rhoc = np.log(r1)
        rlist, sol = [rhoc], [initial]
        for _ in range(ns):
            x = rlist[-1]
            if x <= 0:
                break
            step = min(dl, x)
            sol.append(rk45_solver(self.rfun, x, sol[-1], step, sig=-1, l=l))
            rlist.append(max(0., x-step))
        if rmax > r1:
            rlist_out, sol_out = [rhoc], [initial]
            target = np.log(rmax)
            for _ in range(ns):
                x = rlist_out[-1]
                if x >= target:
                    break
                step = min(dl, target-x)
                sol_out.append(rk45_solver(self.rfun, x, sol_out[-1], step, sig=1, l=l))
                rlist_out.append(min(target, x+step))
            rlist = rlist[::-1]+rlist_out[1:]
            sol = sol[::-1]+sol_out[1:]
        return np.array(rlist), np.array(sol)

    def shooting_method(self, l=0, **kwargs):
        aim_i   = kwargs.get('aim_i', -10)
        aim_f   = kwargs.get('aim_f', 100)
        aim_N   = kwargs.get('aim_N', 100)
        aim_try = np.linspace(aim_i,aim_f,aim_N)
        v0      = self.vout(0)
        trial_shooting = []
        for ss in aim_try:
            initial = np.array([0,10**(-ss)])
            _,sol   = self.time_integration(initial, l=l)
            # trial_shooting.append(sol[-1].sum())
            trial_shooting.append(sol[-1,1]+(1-nu0*v0)*sol[-1,0])
        root = brentq(lambda x: interp1d(aim_try,np.log10(np.abs(trial_shooting)),kind='cubic')(x), aim_i, aim_f)
        initial = np.array([0,10**(-root)])
        rlist,sol = self.time_integration(initial,l=l)
        return rlist, sol

    def compute_coefficient(self, **kwargs):
        t0 = time.time()
        lmax   = kwargs.get('lmax', self.lmax)
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        Alm = {il: {} for il in range(lmax + 1)} # coefficient for sin(m phi)
        Blm = {il: {} for il in range(lmax + 1)} # coefficient for cos(m phi)
        rr,tt,pp = self.get_rtp()
        tt       = torch.from_numpy(tt[0]  ).to(device)
        pp       = torch.from_numpy(pp[0]  ).to(device)
        Br       = torch.from_numpy(self.Br).to(device)
        for m in range(0,lmax+1):
            for l in range(m,lmax+1):
                if l==m:
                    P_l00 = Associated_Legendre(l,  m, torch.cos(tt))
                    P_lp1 = Associated_Legendre(l+1,m, torch.cos(tt))
                    cosmp = torch.cos(m*pp)
                    sinmp = torch.sin(m*pp)
                    if l==0:
                        Alm[l][m] = 0
                        Blm[l][m] = (torch.sum(Br*P_l00*torch.sin(tt))/torch.sum(P_l00**2*torch.sin(tt))).item()
                    else:
                        Alm[l][m] = (torch.sum(Br*P_l00*sinmp*torch.sin(tt))/torch.sum(P_l00**2*sinmp**2*torch.sin(tt))).item()
                        Blm[l][m] = (torch.sum(Br*P_l00*cosmp*torch.sin(tt))/torch.sum(P_l00**2*cosmp**2*torch.sin(tt))).item()
                    P_ln1 = P_l00
                    P_l00 = P_lp1
                else:
                    P_lp1 = Associated_Legendre(l+1, m, torch.cos(tt), pn2=P_ln1, pn1=P_l00)
                    cosmp = torch.cos(m*pp)
                    sinmp = torch.sin(m*pp)
                    Alm[l][m] = (torch.sum(Br*P_l00*sinmp*torch.sin(tt))/torch.sum(P_l00**2*sinmp**2*torch.sin(tt))).item()
                    Blm[l][m] = (torch.sum(Br*P_l00*cosmp*torch.sin(tt))/torch.sum(P_l00**2*cosmp**2*torch.sin(tt))).item()
                    if m==0:
                        Alm[l][m]=0
                    P_ln1 = P_l00
                    P_l00 = P_lp1
        self.Alm = Alm
        self.Blm = Blm
        ti = time.time()
        print(f'Finishing computing coefficient takes {(ti-t0):8.3f} sec...')
        return Alm, Blm

    # def compute_Hl(self,r,l=0,**kwargs):
    #     r_list, sol = self.shooting_method(l=l,**kwargs)
    #     interp_func = interp1d(r_list, sol[:,0], kind='cubic')
    #     Hl          = interp_func(np.log(r))
    #     return Hl

    # def compute_Gl(self, r, l=0, **kwargs):
    #     Hl    = kwargs.get('Hl', self.compute_Hl(r,l=l))
    #     Gl    = np.zeros_like(Hl)
    #     Gl[0] = 1.
    #     for i in range(len(Hl)-1):
    #         Gl[i+1] = 0.5*l*(l+1)*Hl[i]*(1-r[i]**2/r[i+1]**2)+Gl[i]*r[i]**2/r[i+1]**2
    #     return Gl

    # def _single_task(self, rls, l=0):
    #     Hl = self.compute_Hl(rls, l=l)
    #     Gl = self.compute_Gl(rls, l=l, Hl=Hl)
    #     return l, Hl, Gl
    
    def compute_HG(self,r,l=0,**kwargs):
        r = np.asarray(r, dtype=float)
        if l == 0:
            return np.zeros_like(r), 1.0/r**2
        r_list, sol = self.shooting_method(l=l,**kwargs)
        interp_Hl   = interp1d(r_list, sol[:,0], kind='cubic')
        Gsol        = sol[:,1]+(1-nu0*np.exp(r_list)*self.vout(r_list))*sol[:,0]
        interp_Gl   = interp1d(r_list, Gsol    , kind='cubic')
        Hl          = interp_Hl(np.log(r))
        Gl          = interp_Gl(np.log(r))
        return Hl,Gl

    def _single_task(self, rls, l=0):
        Hl,Gl = self.compute_HG(rls,l=l)
        return l, Hl, Gl

    def get_outflow_field(self, **kwargs):
        t0       = time.time()
        device   = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        device   = torch.device(device)
        lmax     = kwargs.get('lmax', self.lmax)
        Alm      = kwargs.get('Alm', None)
        Blm      = kwargs.get('Blm', None)
        n_cores  = kwargs.get('n_cores', 50)
        FM       = kwargs.get('fast_mode', False)
        if FM and os.path.exists('./OFF_temp_files'):
            raise FileExistsError('Refusing pre-existing OFF_temp_files in the current directory')
        OF       = kwargs.get('off_file' , self.off_file)
        if Alm is None or Blm is None:
            Alm,Blm = self.compute_coefficient(**kwargs)
        rr,tt,pp = self.get_rtp()
        rls      = rr[:,0,0]
        tls      = tt[0,:,0]
        pls      = pp[0,0,:]
        Nr,Nt,Np = self.n_r,self.n_t,self.n_p
        rr,tt,pp = torch.from_numpy(np.stack([rr,tt,pp])).to(device)
        br = torch.zeros_like(rr, dtype=torch.float64).to(device)
        bt = torch.zeros_like(tt, dtype=torch.float64).to(device)
        bp = torch.zeros_like(pp, dtype=torch.float64).to(device)
        HL = {il: {} for il in range(lmax+1)}
        GL = {il: {} for il in range(lmax+1)}
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = []
            for l in range(lmax+1):
                futures.append(executor.submit(self._single_task, rls, l))
            for future in as_completed(futures):
                l,hl,gl = future.result()
                HL[l]   = hl
                GL[l]   = gl
        print(f'Complete calculating the radial function Hl and Gl, wall_time: {(time.time()-t0):8.3f} sec...')
        if not FM:
            for m in range(0, lmax+1):
                for l in range(m, lmax+1):
                    alm = Alm[l][m]
                    blm = Blm[l][m]
                    if l==m:
                        P_l00 = Associated_Legendre(l,  m, torch.cos(tt))
                        P_lp1 = Associated_Legendre(l+1,m, torch.cos(tt))
                        cosmp = torch.cos(m*pp)
                        sinmp = torch.sin(m*pp)
                        Hl    = HL[l]
                        Gl    = GL[l]
                        Hl    = Hl[:,np.newaxis,np.newaxis].repeat(Nt,axis=1).repeat(Np,axis=2)
                        Gl    = Gl[:,np.newaxis,np.newaxis].repeat(Nt,axis=1).repeat(Np,axis=2)
                        Hl    = torch.from_numpy(Hl).to(device)
                        Gl    = torch.from_numpy(Gl).to(device)
                        Qlm   = P_l00
                        DQlm  = DAssociated_Legendre(l,m,torch.cos(tt),P_l00=P_l00,P_lp1=P_lp1)
                        Pm1   = sinmp
                        Pm2   = cosmp
                        Dpm1  = m*cosmp
                        Dpm2  =-m*sinmp
                        if l==0:
                            br+=blm*Gl*Qlm*Pm2
                        else:
                            br+=Gl*Qlm*(alm*Pm1+blm*Pm2)
                            bt+=-Hl*DQlm*torch.sin(tt)*(alm*Pm1+blm*Pm2)
                            bp+=Hl/torch.sin(tt)*Qlm*(alm*Dpm1+blm*Dpm2)
                        P_ln1 = P_l00
                        P_l00 = P_lp1
                    else:
                        P_lp1 = Associated_Legendre(l+1, m, torch.cos(tt), pn2=P_ln1, pn1=P_l00)
                        cosmp = torch.cos(m*pp)
                        sinmp = torch.sin(m*pp)
                        Hl    = HL[l]
                        Gl    = GL[l]
                        Hl    = Hl[:,np.newaxis,np.newaxis].repeat(Nt,axis=1).repeat(Np,axis=2)
                        Gl    = Gl[:,np.newaxis,np.newaxis].repeat(Nt,axis=1).repeat(Np,axis=2)
                        Hl    = torch.from_numpy(Hl).to(device)
                        Gl    = torch.from_numpy(Gl).to(device)
                        Qlm   = P_l00
                        DQlm  = DAssociated_Legendre(l,m,torch.cos(tt),P_l00=P_l00,P_lp1=P_lp1)
                        Pm1   = sinmp
                        Pm2   = cosmp
                        Dpm1  = m*cosmp
                        Dpm2  =-m*sinmp
                        br+=Gl*Qlm*(alm*Pm1+blm*Pm2)
                        bt+=-Hl*DQlm*torch.sin(tt)*(alm*Pm1+blm*Pm2)
                        bp+=Hl/torch.sin(tt)*Qlm*(alm*Dpm1+blm*Dpm2)
                        P_ln1 = P_l00
                        P_l00 = P_lp1           
            br  =  np.real(br.detach().cpu().numpy())
            bt  =  np.real(bt.detach().cpu().numpy())
            bp  =  np.real(bp.detach().cpu().numpy())
            ret =  np.stack([br,bt,bp], axis=0)
        else:
            save_name = kwargs.get('save_name', self.save_name)
            devices   = kwargs.get('devices'  , ['cuda:0']    )
            PY        = kwargs.get('python'   , 'python '     )
            self.save_name  = save_name
            self.lmax       = lmax
            self.info['HL'] = HL
            self.info['GL'] = GL
            nD = len(devices)
            di = (lmax+1)*(lmax+2)//2/nD
            m_assigned = [0]
            icnt = 0
            for m in range(lmax+1):
                icnt+=lmax+1-m
                if icnt > di:
                    m_assigned.append(m+1)
                    icnt = icnt % di
            m_assigned.append(lmax+1)
            self.info['mlist']   = m_assigned
            self.info['devices'] = devices
            self.save(save_name=save_name)
            commands = []
            for i,D in enumerate(devices):
                command = PY+f' -u -m pfss.off_scripts -i {save_name} -n {i}'
                commands.append(command)
            processes = []
            for command in commands:
                process = subprocess.Popen(command, shell=True)
                processes.append(process)
            returncodes = [process.wait() for process in processes]
            if any(code != 0 for code in returncodes):
                raise RuntimeError(f"OFF child process failed: return codes {returncodes}")
            off_files = sorted(glob.glob('./OFF_temp_files/*.npy'))
            expected = sorted(f'./OFF_temp_files/off_{m_assigned[i]:03d}_{m_assigned[i+1]:03d}.npy'
                              for i in range(len(devices)))
            if off_files != expected:
                raise RuntimeError(f"OFF child output mismatch: expected {expected}, found {off_files}")
            ret = np.zeros((3,Nr,Nt,Np))
            for f in off_files:
                partial = np.load(f)
                if partial.shape != ret.shape or not np.isfinite(partial).all():
                    raise RuntimeError(f"Invalid OFF child output: {f}")
                ret += partial
            br,bt,bp = ret
            shutil.rmtree('./OFF_temp_files')
    
        self.Br_BB = br[ 0]
        self.Br_SS = br[-1]
        ti = time.time()
        self.off_file = OF
        np.save(OF, ret)
        print(f'Building the Outflow Field successfully using time: {(ti-t0)/60:8.3f} min...')
        return ret

    def save_vts(self, **kwargs):
        vts_name = kwargs.pop('vts_name', 'off')
        Brtp     = kwargs.pop('Brtp', np.load(self.off_file))
        super().save_vts(vts_name=vts_name, Brtp=Brtp, **kwargs)

    def save_vtu(self, **kwargs):
        vtu_name = kwargs.pop('vtu_name', 'off')
        Brtp     = kwargs.pop('Brtp', np.load(self.off_file))
        super().save_vtu(vtu_name=vtu_name, Brtp=Brtp,**kwargs)

    def show_magline(self, **kwargs):
        load_file = self.off_file
        super().show_magline(load_file=load_file, **kwargs)
