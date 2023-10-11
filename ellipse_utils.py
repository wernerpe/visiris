from pydrake.all import Hyperellipsoid, MathematicalProgram, Solve
import numpy as np

def switch_ellipse_description(A, b):
    d = np.linalg.solve(A.T@A, -A.T@b)
    return Hyperellipsoid(A,d), A, d

def get_seed_ellipse(pt1, pt2, eps = 0.01):
    dim = pt1.shape[0]
    pts = [pt1, pt2]
    for _ in range(2*dim):
        m = 0.5*(pt1+pt2) + eps*(np.random.rand(2,1)-0.5)
        pts.append(m)

    prog = MathematicalProgram()
    A = prog.NewSymmetricContinuousVariables(dim, 'A')
    b = prog.NewContinuousVariables(dim, 'b')
    prog.AddMaximizeLogDeterminantCost(A)
    for idx, pt in enumerate(pts):
        S = prog.NewSymmetricContinuousVariables(dim+1, 'S')
        prog.AddPositiveSemidefiniteConstraint(S)
        prog.AddLinearEqualityConstraint(S[0,0] == 1)
        v = (A@pt + b.reshape(2,1)).T
        c = (S[1:,1:]-np.eye(dim)).reshape(-1)
        for idx in range(dim):
            prog.AddLinearEqualityConstraint(S[0,1 + idx]-v[0,idx], 0 )
        for ci in c:
            prog.AddLinearEqualityConstraint(ci, 0 )

    prog.AddPositiveSemidefiniteConstraint(A)

    sol = Solve(prog)
    if sol.is_success():
        return switch_ellipse_description(sol.GetSolution(A), sol.GetSolution(b))
    else:
        return None, None, None

def get_lj_ellipse(pts):
    if len(pts) ==1:
        S = Hyperellipsoid.MakeHypersphere(1e-3, pts[0, :])
        return S#, S.A(), S.center()
    # if len(pts) ==2:
    #     S, _,_ = get_seed_ellipse(pts[0,:].reshape(2,1), pts[1,:].reshape(2,1), eps= 0.01)
    #     return S 
    dim = pts[0].shape[0]
    # pts = #[pt1, pt2]
    # for _ in range(2*dim):
    #     m = 0.5*(pt1+pt2) + eps*(np.random.rand(2,1)-0.5)
    #     pts.append(m)
    prog = MathematicalProgram()
    A = prog.NewSymmetricContinuousVariables(dim, 'A')
    b = prog.NewContinuousVariables(dim, 'b')
    prog.AddMaximizeLogDeterminantCost(A)
    for idx, pt in enumerate(pts):
        pt = pt.reshape(dim,1)
        S = prog.NewSymmetricContinuousVariables(dim+1, 'S')
        prog.AddPositiveSemidefiniteConstraint(S)
        prog.AddLinearEqualityConstraint(S[0,0] == 0.9)
        v = (A@pt + b.reshape(dim,1)).T
        c = (S[1:,1:]-np.eye(dim)).reshape(-1)
        for idx in range(dim):
            prog.AddLinearEqualityConstraint(S[0,1 + idx]-v[0,idx], 0 )
        for ci in c:
            prog.AddLinearEqualityConstraint(ci, 0 )

    prog.AddPositiveSemidefiniteConstraint(A) # eps * identity
    prog.AddPositiveSemidefiniteConstraint(10000*np.eye(dim)-A)

    sol = Solve(prog)
    if sol.is_success():
        HE, _, _ =switch_ellipse_description(sol.GetSolution(A), sol.GetSolution(b))
        return HE
    else:
        return None


def plot_ellipse(ax, H, n_samples, color = None, linewidth = 1):
    A = H.A()
    center = H.center()
    angs = np.linspace(0, 2*np.pi, n_samples+1)
    coords = np.zeros((2, n_samples + 1))
    coords[0, :] = np.cos(angs)
    coords[1, :] = np.sin(angs)
    Bu = np.linalg.inv(A)@coords
    pts = Bu + center.reshape(2,1)
    if color is None:
        ax.plot(pts[0, :], pts[1, :], linewidth = linewidth)
    else:
        ax.plot(pts[0, :], pts[1, :], linewidth = linewidth, color = color)

def plot_ellipse_homogenous_rep(ax, Emat, xrange, yrange, resolution, linewidth =1, color = 'b', zorder = 1,):
    # center = np.linalg.solve(-Emat[:-1, :-1], Emat[-1, :-1])
    # max = np.max(1/(np.sqrt(np.linalg.eigh(Emat[:-1, :-1])[0] + 1e-3)))
    # if clip is None:
    #     clip = [-10,10]
    x = np.arange(xrange[0], xrange[1], resolution)
    y = np.arange(yrange[0], yrange[1], resolution)
    X,Y = np.meshgrid(x,y)
    hE = arrange_homogeneous_ellipse_matrix_to_vector(Emat)
    Z = (hE@build_quadratic_features(np.concatenate((X.reshape(-1,1), Y.reshape(-1,1)), axis = 1)).T).reshape(len(y), len(x))
    CS = ax.contour(X, Y, Z,[1.0], linewidths = linewidth,zorder = zorder, colors = [color])
    return CS

def get_homogeneous_matrix(E):
    ## (q^T 1)@Emat@(q^T 1).T<=1 is the homogeneous form 
    dim = len(E.center())
    Eupp = E.A().T@E.A()
    Eoff = (-E.center().T@E.A().T@E.A()).reshape(1,-1)
    Econst = E.center().T@E.A().T@E.A()@E.center()
    Emat = np.zeros((dim+1, dim +1))
    Emat[:dim, :dim] = Eupp
    Emat[:dim, -1] = Eoff.squeeze()
    Emat[-1, :dim] = Eoff.squeeze()
    Emat[-1,-1] = Econst
    return Emat

def get_hyperellipsoid_from_homogeneous_matrix(Emat):
    An = (np.linalg.cholesky(Emat[:-1, :-1])).T
    center = np.linalg.solve(-Emat[:-1, :-1], Emat[-1, :-1])
    return Hyperellipsoid(An, center)

def arrange_homogeneous_ellipse_matrix_to_vector(Emat):
    #flattens the homogenous matrix describing the ellipsoid to a vector
    # HE  = (diagonal, offdiagonal terms *2 row first ) 
    dim = Emat.shape[0]
    hE = []
    #hE = np.zeros(int(dim*(dim+1)/2))
    index = 0
    for i in range(dim):
        #hE[index] = Emat[i,i]
        hE.append(1*Emat[i,i])
        index+=1
    for i in range(dim):
        for j in range(i+1, dim):
            #hE[index] = 2*Emat[i,j]
            hE.append(2*Emat[i,j])
            index+=1
    return np.array(hE)

def build_quadratic_features(q_mat):
    #input are row-wise points in Rn
    dim = q_mat.shape[1]+1
    q_mat_hom = np.concatenate((q_mat, np.ones((q_mat.shape[0],1))), axis =1)
    num_features = int(dim*(dim+1)/2)
    features = np.zeros((q_mat.shape[0], num_features))
    index = 0
    #quadratic features
    for i in range(dim):
        features[:, index] = q_mat_hom[:, i]*q_mat_hom[:, i]
        index+=1
    #mixed features
    for i in range(dim-1):
        for j in range(i+1, dim):
            features[:, index] = q_mat_hom[:,i]*q_mat_hom[:,j]
            index+=1
    return features