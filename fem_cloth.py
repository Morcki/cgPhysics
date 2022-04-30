import taichi as ti

ti.init(arch=ti.cuda)
# Screen Setting
res_w = 1080
res_h = 720
pixels = ti.field(dtype=float, shape=(res_w, res_h))
# Cloth Model Setting
w = 11
h = 11
padding = 30
nparticles = w * h
ntriangles = (w-1) * (h-1) * 2

# State Equation Setting
isPaused    = ti.field(dtype=ti.i32, shape=())
dt          = 0.0005
damping     = 1.5
# ref:https://en.wikipedia.org/wiki/Neo-Hookean_solid
E, nu = 4e4, 0.2  # Young's modulus and Poisson's ratio
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)  # Lame parameters
internal_force_aux_scalar = 1 / 8

external    = ti.Vector.field(n = 2, dtype = ti.f32, shape = 1) # ti.Vector([0.0,  0.0])
gravity     = ti.Vector.field(n = 2, dtype = ti.f32, shape = 1) # ti.Vector([0.0, -9.8])
niterator   = 30

# Interactive Setting
isMouseLeftButtonHandle = ti.field(dtype=ti.i32, shape=())
mouse_last_x = ti.field(dtype=ti.f32, shape=())
mouse_last_y = ti.field(dtype=ti.f32, shape=())
gui = ti.GUI('Position-based Dynamics Demo', res = (res_w, res_h), background_color = 0xDDDDDD)

############################## Taichi Var Define ##############################
cloth_tri  = ti.field(ti.i32, shape = (ntriangles    , 3))
cloth_edge = ti.field(ti.i32, shape = (ntriangles * 3, 2))

cloth_pos  = ti.Vector.field(n = 2, dtype = ti.f32, shape = nparticles)
cloth_vel  = ti.Vector.field(n = 2, dtype = ti.f32, shape = nparticles)


dm_inv     = ti.Matrix.field(n = 2, m = 2, dtype = ti.f32, shape = ntriangles)
tri_area   = ti.field(ti.f32, shape = ntriangles)

# dump variable
particle_force = ti.Vector.field(n = 2, dtype = ti.f32, shape = nparticles)
pos_old = ti.Vector.field(n = 2, dtype = ti.f32, shape = nparticles)
############################## Taichi Var Define ##############################


@ti.func
def build_split03_quad(index : ti.i32, i : ti.i32, j : ti.i32):
    # 4____2/3
    # |    /|
    # |   / |
    # |  /  |
    # | /   |
    # |/____|
    # 0/5   1
    
    cloth_tri[index    , 0] = j * w + i
    cloth_tri[index    , 1] = j * w + i + 1
    cloth_tri[index    , 2] = (j + 1) * w + i + 1
    
    cloth_tri[index + 1, 0] = (j + 1) * w + i + 1
    cloth_tri[index + 1, 1] = (j + 1) * w + i
    cloth_tri[index + 1, 2] = j * w + i

@ti.func    
def build_split12_quad(index : ti.i32, i : ti.i32, j : ti.i32):
    #  _____
    # |\    |
    # | \   |
    # |  \  |
    # |   \ |
    # |____\|
    #
    
    cloth_tri[index    , 0] = j * w + i
    cloth_tri[index    , 1] = j * w + i + 1
    cloth_tri[index    , 2] = (j + 1) * w + i
    
    cloth_tri[index + 1, 0] = (j + 1) * w + i
    cloth_tri[index + 1, 1] = (j + 1) * w + i + 1
    cloth_tri[index + 1, 2] = j * w + i + 1
    
@ti.func
def calc_dm_inv(index : ti.i32):
    # calc inverse of Dm
    v0 = cloth_tri[index, 0]
    v1 = cloth_tri[index, 1]
    v2 = cloth_tri[index, 2]
    
    X10 = cloth_pos[v1] - cloth_pos[v0]
    X20 = cloth_pos[v2] - cloth_pos[v0]
    
    dm_inv[index] = ti.Matrix.cols([X10, X20]).inverse()
    # calc triangle original area
    tri_area[index] = 0.5 * abs(X10.cross(X20))
    
@ti.func
def para_init_cloth_status():
    wlength = w - 1
    hlength = h - 1
    for i in cloth_pos:
        cloth_pos[i] = ti.Vector([-1 / 2 + (i % w) / wlength, -1 / 2 + (i // w) / hlength])
        cloth_vel[i] = ti.Vector([0.0, 0.0])
        
@ti.func
def para_init_cloth_triangle():
    for i, j in ti.ndrange(w - 1, h - 1):
        t = 2 * (i + j * (w - 1))
        if (i % 2 == j % 2):
            build_split03_quad(t, i, j)
        else:
            build_split12_quad(t, i, j)
    
    for i, j in cloth_tri:
        cloth_edge[i * 3 + j, 0] = cloth_tri[i, j]
        cloth_edge[i * 3 + j, 1] = cloth_tri[i, (j + 1) % 3]
    
    for i in range(ntriangles):
        calc_dm_inv(i)
        

@ti.kernel
def build_cloth_model():
    para_init_cloth_status()
    para_init_cloth_triangle()
    
@ti.kernel
def init_force_params():
    external[0] = ti.Vector([0.0,  0.0])
    gravity[0]  = ti.Vector([0.0, -9.8])

@ti.func
def fix_cloth_point():
    # Fix Two Point 
    cloth_vel[nparticles - w] += (pos_old[nparticles - w] - cloth_pos[nparticles - w]) / dt
    cloth_vel[nparticles - 1] += (pos_old[nparticles - 1] - cloth_pos[nparticles - 1]) / dt
    
    cloth_pos[nparticles - w] = pos_old[nparticles - w]
    cloth_pos[nparticles - 1] = pos_old[nparticles - 1]
    

@ti.kernel
def para_epoch_shear_cloth():
    for i in cloth_pos:
        if ((i == nparticles - 1) or (i == nparticles - w)):
            continue
        cloth_pos[i][0] += 1

@ti.func
def para_epoch_init_force():
    for i in particle_force:
        particle_force[i] = ti.Vector([0.0,  0.0])

def update_external_force(fx : ti.f32, fy : ti.f32):
    external[0] = ti.Vector([fx, fy])
    
@ti.func
def para_epoch_fem_kernel():
    for i in range(ntriangles):
        v0 = cloth_tri[i, 0]
        v1 = cloth_tri[i, 1]
        v2 = cloth_tri[i, 2]
        
        x10 = cloth_pos[v1] - cloth_pos[v0]
        x20 = cloth_pos[v2] - cloth_pos[v0]
        
        F = ti.Matrix.cols([x10, x20]) @ dm_inv[i]
        
        # potential energy of each face (Neo-Hookean)
        F_it =  F.inverse().transpose()                              # Important
        PF = mu * (F - F_it) + lam * ti.log(F.determinant()) * F_it  # Important
        H = -tri_area[i]* PF @ dm_inv[i].transpose()                 # Important
        
        f1 = ti.Vector([H[0, 0], H[1, 0]])
        f2 = ti.Vector([H[0, 1], H[1, 1]])
        f0 = -(f1 + f2)
        
        particle_force[v0] += f0
        particle_force[v1] += f1
        particle_force[v2] += f2

@ti.func
def para_update_status():
    for i in cloth_pos:
        inter_force = particle_force[i] * internal_force_aux_scalar + gravity[0]
        cloth_vel[i] += (inter_force + external[0]) * dt
        cloth_vel[i] *= ti.exp(-dt * damping)
        cloth_pos[i] += cloth_vel[i] * dt
    
    fix_cloth_point()

@ti.kernel
def simulator_kernel():
    para_epoch_init_force()
    para_epoch_fem_kernel()
    para_update_status()

def init_simulator():
    build_cloth_model()
    init_force_params()
    
    pos_old.copy_from(cloth_pos)
    
def run_simulator():
    for i in range(niterator):
        simulator_kernel()
        
    # debug_arr = external.to_numpy()
    # print(debug_arr)
    
    
def draw_cloth_mesh_to_screen():
    pos = cloth_pos.to_numpy()
    # scale transform
    pos[:,0] *= padding * (w - 1)
    pos[:,1] *= padding * (h - 1)
    # translation transform
    pos[:,0] += res_w / 2
    pos[:,1] += res_h / 2
    # projection to screen
    pos[:,0] /= res_w
    pos[:,1] /= res_h
    
    edges_np = cloth_edge.to_numpy()
    
    X = pos[edges_np[:, 0]]
    Y = pos[edges_np[:, 1]]
    gui.lines(begin = X, end = Y, radius = 2, color = 0x068587)
    gui.circles(pos, radius = 5, color = 0xED553B)
    
def handle_mouse_event():
    if not isMouseLeftButtonHandle[None]:
        isMouseLeftButtonHandle[None] = 1
        mouse_last_x[None], mouse_last_y[None] = gui.get_cursor_pos()
    else:
        mouse_x, mouse_y = gui.get_cursor_pos()
        update_external_force(
            10 * (mouse_x - mouse_last_x[None]), 
            10 * (mouse_y - mouse_last_y[None]))
        
        
def main():
    
    init_simulator()
    
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == gui.SPACE:
                isPaused[None] = not isPaused[None]
                
        if gui.is_pressed(ti.GUI.LMB):
            handle_mouse_event()
        else:
            update_external_force(0, 0)
            isMouseLeftButtonHandle[None] = 0
            
        if not isPaused[None]:
            run_simulator()
        
        draw_cloth_mesh_to_screen()
        gui.show()

if __name__ == '__main__':
    main()