'''
Code       : scs_module.py
Date       : 2024.10.15
Contributer: H.Y.Li (liyuhua0909@126.com), G.Y.Chen (gychen@smail.nju.edu.cn)
Purpose    : Extending the PFSS model to a larger scale...

### --------------------------------- ###
Remark:
2024.10.15: Build the code
'''

from .needs import *

from .pfss_module import pfss_solver
from .funcs import brtp2bxyz, trilinear_interpolation, Brtp_lm, Associated_Legendre
from .magline import rk45, magline_stepper, magline_solver, show_boundary, show_maglines,parallel_magline_solver,show_current_sheet

# ================

def reorientation(Br_cp,Bt_cp,Bp_cp):
    mask_negative = Br_cp < 0
    Br_cp[mask_negative] = -Br_cp[mask_negative]
    Bt_cp[mask_negative] = -Bt_cp[mask_negative]
    Bp_cp[mask_negative] = -Bp_cp[mask_negative]
    return Br_cp, Bt_cp, Bp_cp

def Pnm(n, m, theta, **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    is_array = isinstance(theta, np.ndarray)
    if is_array:
        theta = torch.from_numpy(theta).to(device)
    delta = 0 if m!=0 else 1
    ret   = np.sqrt((2-delta)*float(math.factorial(n-m))/float(math.factorial(n+m)))
    ret   = ret*Associated_Legendre(n,m, torch.cos(theta), **kwargs)
    if is_array:
        return ret.detach().cpu().numpy()
    else:
        return ret

def DPnm(n, m, theta, **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    is_array = isinstance(theta, np.ndarray)
    if is_array:
        theta = torch.from_numpy(theta). to(device)
    P_lp0_m = kwargs.get('P_lp0_m', Associated_Legendre(n  ,m,torch.cos(theta),**kwargs))
    P_lp1_m = kwargs.get('P_lp1_m', Associated_Legendre(n+1,m,torch.cos(theta),**kwargs))
    dL_dth = 1/torch.sin(theta)*(-(n+1)*torch.cos(theta)*P_lp0_m+(n-m+1)*P_lp1_m)
    delta = 0 if m!=0 else 1
    ret   = np.sqrt((2-delta)*float(math.factorial(n-m))/float(math.factorial(n+m)))
    ret   = ret*dL_dth
    if is_array:
        return ret.detach().cpu().numpy()
    else:
        return ret

def alpha_beta(n,m,tt,pp,**kwargs):
    device    = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device    = torch.device(device)
    is_array  = isinstance(tt, np.ndarray) or isinstance(pp, np.ndarray)
    if is_array:
        tt    = torch.from_numpy(tt).to(device)
        pp    = torch.from_numpy(pp).to(device)
        
    P         =  Pnm(n,m,tt,**kwargs)
    dP_dth    = DPnm(n,m,tt,**kwargs)
    
    alpha_1nm = (n+1)*P*torch.cos(m*pp)
    alpha_2nm = -dP_dth*torch.cos(m*pp)
    alpha_3nm = m/torch.sin(tt)*P*torch.sin(m*pp)

    beta_1nm  = (n+1)*P*torch.sin(m*pp)
    beta_2nm  = -dP_dth*torch.sin(m*pp)
    beta_3nm  = -m/torch.sin(tt)*P*torch.cos(m*pp)

    alpha     = torch.stack([alpha_1nm,alpha_2nm,alpha_3nm], dim=0)
    beta      = torch.stack([beta_1nm ,beta_2nm ,beta_3nm ], dim=0)
    if is_array:
        alpha = alpha.detach().cpu().numpy()
        beta  =  beta.detach().cpu().numpy()
    return alpha, beta

def get_alpha_beta_mat(th_list, ph_list, lmax=80, **kwargs):
    TT,PP = np.meshgrid(th_list, ph_list, indexing='ij')
    th,ph = TT.flatten(), PP.flatten()
    lm_list = [[il,im] for il in range(lmax+1) for im in range(il+1)]
    ret = []
    for l,m in lm_list:
        [alpha1,alpha2,alpha3],[beta1,beta2,beta3] = alpha_beta(l,m,th,ph, **kwargs)
        ret.append(np.hstack([alpha1,alpha2,alpha3]))
    lm_list = [[il,im] for il in range(1, lmax+1) for im in range(1,il+1)]
    for l,m in lm_list:
        [alpha1,alpha2,alpha3],[beta1,beta2,beta3] = alpha_beta(l,m,th,ph, **kwargs)
        ret.append(np.hstack([beta1,beta2,beta3]))
    ret = np.stack(ret, axis=0)
    return ret

def build_SCS_Brtp(rr,tt,pp,glm,hlm,lmax=10, **kwargs):
    device    = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device    = torch.device(device)
    Rcp       = kwargs.get('Rcp', 2.49)
    is_array = isinstance(rr, np.ndarray)
    if is_array:
        rr = torch.from_numpy(rr).to(device)
        pp = torch.from_numpy(pp).to(device)
        tt = torch.from_numpy(tt).to(device)
    br = torch.zeros_like(rr)
    bt = torch.zeros_like(tt)
    bp = torch.zeros_like(pp)
    for l in range(lmax+1):
        for m in range(l+1):
            plm  =  Pnm(l,m,tt)
            dplm = DPnm(l,m,tt)
            br+=(l+1)*(Rcp/rr)**(l+2)*plm*(glm[l][m]*torch.cos(m*pp)+hlm[l][m]*torch.sin(m*pp))
            bt+=-(Rcp/rr)**(l+2)*dplm*(glm[l][m]*torch.cos(m*pp)+hlm[l][m]*torch.sin(m*pp))
            bp+=(Rcp/rr)**(l+2)*plm*m/torch.sin(tt)*(glm[l][m]*torch.sin(m*pp)-hlm[l][m]*torch.cos(m*pp))
    Brtp = torch.stack([br,bt,bp],dim=0)
    if is_array:
        Brtp = Brtp.detach().cpu().numpy()
    return Brtp

def _pfss_weight_smooth(r, split_r, half_width):
    """Return PFSS blend weight in [split_r-half_width, split_r+half_width]."""
    if half_width <= 0:
        return (np.asarray(r) <= split_r).astype(float)
    lo = split_r - half_width
    hi = split_r + half_width
    t = (np.asarray(r, dtype=np.float64) - lo) / (hi - lo)
    t = np.clip(t, 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)  # smoothstep
    return 1.0 - s

# ================

class scs_solver(pfss_solver):
    def __init__(self,
                 fits_file,
                 n_r      = 400,
                 n_t      = 200,
                 n_p      = 400,
                 lmax     = 80,
                 Rs       = 2.5,
                 Rcp      = 2.4,
                 Rtp      = 10.,
                 lmax_scs = 10,
                 Nrtp_scs = [200,200,400],
                 **kwargs
                ):
        super().__init__(fits_file,n_r,n_t,n_p,lmax,Rs)
        self.Rcp       = Rcp
        self.Rtp       = Rtp
        self.lmax_scs  = lmax_scs
        self.glm       = None
        self.hlm       = None
        self.Nrtp_scs  = Nrtp_scs
        self.mask      = None
        self.scs_file  = './Brtp_scs.npy'
        self.save_name = 'scs_solver.pkl'
        self.Alm = kwargs.get('Alm', self.Alm)
        self.Blm = kwargs.get('Blm', self.Blm)
        self._initialization(**kwargs)

    def _initialization(self,**kwargs):
        lmax  = self.lmax_scs
        Nt,Np = self.Nrtp_scs[1:]
        Rcp   = self.Rcp
        cusp_method = kwargs.get('cusp_method', 'harmonics')
        dth   = np.pi/Nt
        dph   = np.pi/Np*2
        t_list   = np.linspace(0,  np.pi,Nt+1)[:-1]+0.5*dth
        p_list   = np.linspace(0,2*np.pi,Np+1)[:-1]+0.5*dph
        t_list   = t_list[::-1]
        Tcp,Pcp  = np.meshgrid(t_list,p_list,indexing='ij')
        rr,tt,pp = np.meshgrid(np.array([Rcp]), t_list, p_list, indexing='ij')
        rtp_cp   = np.stack([rr,tt,pp], axis=0)
        Brtp_cp  = kwargs.get('Brtp_cusp', None)
        if Brtp_cp is None:
            # Build the cusp field from the same PFSS representation used later for
            # stitched harmonics evaluation, instead of mixing PFSS interpolation
            # here with PFSS harmonics below the interface.
            Brtp_cp = super().get_Brtp(rtp_cp, method=cusp_method, **kwargs)
        Br_cp,Bt_cp,Bp_cp = Brtp_cp[:,0,:,:].copy()
        self.mask= Br_cp<0
        Br_cp,Bt_cp,Bp_cp = reorientation(Br_cp, Bt_cp, Bp_cp)
        alpha_beta_mat = get_alpha_beta_mat(t_list, p_list, lmax=lmax)
        AB_mat = np.matmul(alpha_beta_mat, alpha_beta_mat.T)
        B_hat  = np.hstack([Br_cp.flatten(),Bt_cp.flatten(),Bp_cp.flatten()])
        GH_hat = np.matmul(np.linalg.inv(AB_mat),np.matmul(alpha_beta_mat,B_hat))
        glm = {il: {} for il in range(lmax + 1)}
        hlm = {il: {} for il in range(lmax + 1)}
        G   = GH_hat[:(lmax+1)*(lmax+2)//2]
        H   = GH_hat[(lmax+1)*(lmax+2)//2:]
        for l in range(lmax+1):
            for m in range(l+1):
                glm[l][m] = G[(l+1)*l//2+m]
        for l in range(lmax+1):
            hlm[l][0]=0
            for m in range(1,l+1):
                hlm[l][m] = H[l*(l-1)//2+m-1]
        self.glm = glm
        self.hlm = hlm

    def _inherited_from_pfss(self, ps):
        for key, value in ps.__dict__.items():
            if key in self.__dict__:
                self.__dict__[key] = value

    def get_rtp(self,**kwargs):
        Nrtp = kwargs.get('Nrtp', self.Nrtp_scs)
        Nr,Nt,Np = Nrtp
        dth = np.pi/Nt
        dph = np.pi/Np*2
        t_list = np.linspace(np.pi,0,Nt+1)[1:]+0.5*dth
        p_list = np.linspace(0,2*np.pi,Np+1)[:-1]+0.5*dph
        r_list = np.linspace(self.Rcp,self.Rtp,Nr)
        rr,tt,pp = np.meshgrid(r_list,t_list,p_list,indexing='ij')
        return np.stack([rr,tt,pp])

    def get_scs(self, **kwargs):
        print('Start to build the SCS field...')
        t0        = time.time()
        fname     = kwargs.get('fname', self.scs_file)
        lmax      = kwargs.pop('lmax', self.lmax_scs)
        glm       = self.glm
        hlm       = self.hlm
        rr,tt,pp  = self.get_rtp()
        Nr,Nt,Np  = self.Nrtp_scs
        kwargs.setdefault('Rcp', self.Rcp)
        Br,Bt,Bp  = build_SCS_Brtp(rr,tt,pp,glm,hlm,lmax=lmax,**kwargs)
        mask      = self.mask[np.newaxis,:,:].repeat(Nr,axis=0)
        ret       = np.stack([Br,Bt,Bp])
        ret[:,mask] = -ret[:,mask]
        self.scs_file = fname
        np.save(fname, ret)
        print(f'Finishing calculation takes {(time.time()-t0)/60:8.3f} min...')
        return ret

    def plot_cusp(self, **kwargs):
        Brtp = np.load(self.scs_file)
        Br   = Brtp[0][0]
        title = kwargs.pop('title',rf'Cusp Surface $B_r$ at {self.Rcp:.2f} $R_\odot$')
        self.plot(Br,title=title,**kwargs)

    def save_vts(self, **kwargs):
        vts_name = kwargs.pop('vts_name', 'scs')
        Brtp     = kwargs.pop('Brtp', np.load(self.scs_file))
        super().save_vts(vts_name=vts_name, Brtp=Brtp, **kwargs)

    def save_vtu(self, **kwargs):
        vtu_name = kwargs.pop('vtu_name', 'scs')
        Brtp     = kwargs.pop('Brtp', np.load(self.scs_file))
        super().save_vtu(vtu_name=vtu_name, Brtp=Brtp,**kwargs)

    def load_Brtp(self,**kwargs):
        Brtp = np.load(self.scs_file)
        return Brtp

    def _get_mask_at(self, tt, pp):
        """Get polarity mask values at arbitrary (theta, phi) via nearest-neighbor lookup.

        The mask is defined on the cusp-surface angular grid (Nt x Np).
        This method maps arbitrary (theta, phi) to the nearest grid cell
        and returns the stored polarity flag.
        """
        Nt, Np = self.mask.shape
        dth = np.pi / Nt
        dph = 2 * np.pi / Np
        if isinstance(tt, np.ndarray):
            it = np.clip(np.round((np.pi - tt) / dth - 0.5).astype(int), 0, Nt - 1)
            ip = np.round(pp / dph - 0.5).astype(int) % Np
            return self.mask[it, ip]
        else:
            it = int(np.clip(round((np.pi - tt) / dth - 0.5), 0, Nt - 1))
            ip = int(round(pp / dph - 0.5)) % Np
            return self.mask[it, ip]

    def get_Brtp(self, rtp, **kwargs):
        """
        Get the magnetic field at arbitrary (r, theta, phi) positions.

        Parameters:
            rtp          : spherical coordinates [r, theta, phi]
            method (str) : 'interpolation' uses pre-computed grid data (default);
                           'harmonics' evaluates directly from spherical harmonic
                           coefficients (Alm/Blm for PFSS, glm/hlm for SCS),
                           no pre-computed grid files needed.
            device (str) : torch device, used by 'harmonics' method.
            sc_split_radius (float, optional): radius where PFSS ends and SCS starts.
                Default ``None`` uses ``self.Rs`` (PFSS source surface, i.e. R_pfss):
                PFSS for r <= Rs, SCS for r > Rs. The SCS multipole expansion still uses
                ``Rcp`` (cusp / R_hat) inside ``build_SCS_Brtp``. Pass ``self.Rcp`` to
                restore the previous behaviour (split at the cusp surface).
            interface_blend_half_width (float, optional): if > 0, use smooth blending
                around ``sc_split_radius`` for harmonics evaluation. In the transition
                layer [split_r-w, split_r+w], return
                w_pfss * B_pfss + (1-w_pfss) * B_scs to suppress interface jumps.
        """
        rtp      = np.asarray(rtp, dtype=float)
        scalar = rtp.ndim == 1
        if scalar:
            return self.get_Brtp(rtp[:, None], **kwargs)[:, 0]
        method   = kwargs.pop('method', 'interpolation')
        _split = kwargs.pop('sc_split_radius', None)
        split_r = float(self.Rs if _split is None else _split)
        _bw = kwargs.pop('interface_blend_half_width', 0.0)
        blend_hw = 0.0 if _bw is None else float(_bw)
        r, t, p  = rtp

        # ===== harmonics: direct evaluation from coefficients =====
        if method == 'harmonics':
            def _eval_scs(rr_u, tt_u, pp_u):
                build_kw = {'Rcp': self.Rcp}
                if 'device' in kwargs:
                    build_kw['device'] = kwargs['device']
                ret_s = build_SCS_Brtp(rr_u, tt_u, pp_u, self.glm, self.hlm,
                                       lmax=self.lmax_scs, **build_kw)
                mask_s = self._get_mask_at(tt_u, pp_u)
                ret_s[:, mask_s] = -ret_s[:, mask_s]
                return ret_s

            if isinstance(r, np.ndarray):
                ret = np.zeros_like(rtp)
                if blend_hw > 0:
                    r_lo = split_r - blend_hw
                    r_hi = split_r + blend_hw
                    region_pfss = r < r_lo
                    region_scs = r > r_hi
                    region_blend = ~(region_pfss | region_scs)
                else:
                    region_scs = r > split_r
                    region_pfss = ~region_scs
                    region_blend = np.zeros_like(r, dtype=bool)

                if np.any(region_pfss):
                    rtp_lower = rtp[:, region_pfss]
                    ret_lower = super().get_Brtp(rtp_lower, method='harmonics', **kwargs)
                    ret[:, region_pfss] = ret_lower

                if np.any(region_scs):
                    rtp_upper            = rtp[:, region_scs]
                    rr_u, tt_u, pp_u     = rtp_upper
                    ret_upper = _eval_scs(rr_u, tt_u, pp_u)
                    ret[:, region_scs] = ret_upper

                if np.any(region_blend):
                    rtp_mid = rtp[:, region_blend]
                    rr_m, tt_m, pp_m = rtp_mid
                    ret_pfss = super().get_Brtp(rtp_mid, method='harmonics', **kwargs)
                    ret_scs = _eval_scs(rr_m, tt_m, pp_m)
                    w_pfss = _pfss_weight_smooth(rr_m, split_r, blend_hw)[np.newaxis, ...]
                    ret[:, region_blend] = w_pfss * ret_pfss + (1.0 - w_pfss) * ret_scs

            elif isinstance(r, (int, float, complex)):
                r0 = float(r)
                if blend_hw > 0 and (split_r - blend_hw) <= r0 <= (split_r + blend_hw):
                    rr_s = np.array([r])
                    tt_s = np.array([t])
                    pp_s = np.array([p])
                    rtp_s = np.stack([rr_s, tt_s, pp_s], axis=0)
                    ret_pfss = super().get_Brtp(rtp_s, method='harmonics', **kwargs)
                    ret_scs = _eval_scs(rr_s, tt_s, pp_s)
                    w_pfss = _pfss_weight_smooth(r0, split_r, blend_hw)
                    ret = (w_pfss * ret_pfss + (1.0 - w_pfss) * ret_scs)[:, 0]
                elif r0 <= split_r:
                    ret = super().get_Brtp(rtp, method='harmonics', **kwargs)
                else:
                    rr_s     = np.array([r])
                    tt_s     = np.array([t])
                    pp_s     = np.array([p])
                    ret = _eval_scs(rr_s, tt_s, pp_s)
                    ret = ret[:, 0]
            else:
                raise ValueError('params `rtp` should be a numpy array or Scalar...')
            return ret

        # ===== interpolation: lookup from pre-computed grid data =====
        Bfile    = kwargs.pop('load_file', self.scs_file)
        Brtp     = np.load(Bfile)
        pfss     = np.load(kwargs.get('pfss_file', self.pfss_file))
        Nr,Nt,Np = self.Nrtp_scs
        if isinstance(r,np.ndarray):
            region_scs = r > split_r
            region_pfss = ~region_scs
            rtp_lower   = rtp[:,region_pfss]
            rtp_upper   = rtp[:,region_scs]
            rr,tt,pp    = rtp_lower
            Nr,Nt,Np    = pfss.shape[1:]
            ir          = (rr-1)/(self.Rs-1)*(Nr-1)
            it          = (np.pi-tt)/np.pi*(Nt-1)
            ip          = (pp-0)/2/np.pi*(Np-1)
            size        = rr.shape
            ir          = ir.flatten()
            it          = it.flatten()
            ip          = ip.flatten()
            idx         = np.stack([ir,it,ip], axis=1)
            ret_lower   = trilinear_interpolation(pfss.transpose(1,2,3,0), idx).T
            ret_lower   = ret_lower.reshape(3,*size)
            Nr,Nt,Np    = self.Nrtp_scs
            rr,tt,pp    = rtp_upper
            ir          = (rr-self.Rcp)/(self.Rtp-self.Rcp)*(Nr-1)
            it          = (np.pi-tt)/np.pi*(Nt-1)
            ip          = (pp-0)/2/np.pi*(Np-1)
            size        = rr.shape
            ir          = ir.flatten()
            it          = it.flatten()
            ip          = ip.flatten()
            idx         = np.stack([ir,it,ip], axis=1)
            ret_upper   = trilinear_interpolation(Brtp.transpose(1,2,3,0), idx).T
            ret_upper   = ret_upper.reshape(3, *size)
            ret         = np.zeros_like(rtp)
            ret[:,region_pfss] = ret_lower
            ret[:,region_scs]  = ret_upper
        elif isinstance(r, (int, float, complex)):
            if float(r) <= split_r:
                ret = super().get_Brtp(rtp)
            else:
                ir  = (r-self.Rcp)/(self.Rtp-self.Rcp)*(Nr-1)
                it  = (np.pi-t)/np.pi*(Nt-1)
                ip  = (p-0)/2/np.pi*(Np-1)
                idx = [ir,it,ip]
                ret = trilinear_interpolation(Brtp.transpose(1,2,3,0), idx)
        else:
            raise ValueError('params `rtp` should have be a numpy array or Scalar...')
        return ret

    def show_maglines(self,**kwargs):
        Rs = kwargs.pop('Rs', self.Rtp)
        super().show_maglines(Rs=Rs, **kwargs)

    def parallel_magline_solver(self, rtps,**kwargs):
        return parallel_magline_solver(self, rtps, **kwargs)

    def show_current_sheet(self, **kwargs):
        return show_current_sheet(self, **kwargs)
