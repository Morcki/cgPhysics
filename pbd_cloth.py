import numpy as np
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

# State Equation Setting
isPaused  = ti.field(dtype=ti.i32, shape=())
dt        = 0.05
alpha     = 0.2
dumping   = 0.98
external  = ti.Vector.field(n = 2, dtype = ti.f32, shape = 1) # ti.Vector([0.0,  0.0])
gravity   = ti.Vector.field(n = 2, dtype = ti.f32, shape = 1) # ti.Vector([0.0, -9.8])
niterator = 32

# Interactive Setting
isMouseLeftButtonHandle = ti.field(dtype=ti.i32, shape=())
mouse_last_x = ti.field(dtype=ti.f32, shape=())
mouse_last_y = ti.field(dtype=ti.f32, shape=())
gui = ti.GUI('Position-based Dynamics Demo', res = (res_w, res_h), background_color = 0xDDDDDD)

############################## Python Scope Start #############################
def build_cloth_edges():
    tri = np.zeros((w-1) * (h-1) * 6)
    t = 0
    for j in range(h - 1):
        for i in range(w - 1):
            tri[t * 6 + 0] = j * w  + i
            tri[t * 6 + 1] = (j + 1) * w  + i
            tri[t * 6 + 2] = (j + 1) * w  + i + 1
            tri[t * 6 + 3] = (j + 1) * w  + i 
            tri[t * 6 + 4] = (j + 1) * w  + i + 1 
            tri[t * 6 + 5] = j * w  + i + 1
            t += 1
    edges_ = np.zeros((len(tri), 2))
    for i in range(0, len(tri), 3):
        edges_[i    ] = [tri[i + 0], tri[i + 1]] if (tri[i + 0] < tri[i + 1]) else [tri[i + 1], tri[i + 0]]
        edges_[i + 1] = [tri[i + 1], tri[i + 2]] if (tri[i + 1] < tri[i + 2]) else [tri[i + 2], tri[i + 1]]
        edges_[i + 2] = [tri[i + 2], tri[i + 0]] if (tri[i + 2] < tri[i + 0]) else [tri[i + 0], tri[i + 2]]
        
    edges_ = edges_[np.argsort(edges_[:,0])]
    first_edge_ind = edges_[0, 0]
    second_edge_set = set()
    second_edge_set.add(edges_[0, 1])
    
    pick_edges = [[edges_[0, 0], edges_[0, 1]]]
    
    for edge in edges_:
        if (edge[0] == first_edge_ind):
            if (edge[1] in second_edge_set):
                continue
            pick_edges.append([edge[0], edge[1]])
            second_edge_set.add(edge[1])
        else:
            first_edge_ind = edge[0]
            second_edge_set.clear()
            second_edge_set.add(edge[1])
            pick_edges.append([edge[0], edge[1]])
    
    for i in range(w - 1):
        pick_edges.append([i, i + 1])
    pick_edges = np.array(pick_edges, dtype=int)
    return pick_edges

edges_np = build_cloth_edges()
############################## Python Scope End ###############################

############################## Taichi Var Define ##############################
nedges = len(edges_np)
cloth_pos  = ti.Vector.field(n = 2, dtype = ti.f32, shape = nparticles)
cloth_vel  = ti.Vector.field(n = 2, dtype = ti.f32, shape = nparticles)
cloth_edge = ti.Vector.field(n = 2, dtype = ti.f32, shape = nedges)
cloth_edge.from_numpy(edges_np)


rest_length = ti.field(ti.f32, shape = nedges)
# dump variable
pos_old = ti.Vector.field(n = 2, dtype = ti.f32, shape = nparticles)
pos_tmp = ti.Vector.field(n = 2, dtype = ti.f32, shape = 1)
pos_new = ti.Vector.field(n = 2, dtype = ti.f32, shape = nparticles)
sum_num = ti.field(ti.f32, shape = nparticles)
############################## Taichi Var Define ##############################

@ti.func
def para_init_cloth():
    wlength = w - 1
    hlength = h - 1
    for i in cloth_pos:
        cloth_pos[i][0] = -wlength / 2 + (i  % w)
        cloth_pos[i][1] = -hlength / 2 + (i // w)
        cloth_vel[i][0] = 0.0
        cloth_vel[i][1] = 0.0

@ti.func
def para_init_restlength():
    for i in rest_length:
        v0 = cloth_edge[i][0]
        v1 = cloth_edge[i][1]
        dx = cloth_pos[v0] - cloth_pos[v1]
        rest_length[i] = dx.norm()
        
@ti.kernel
def init_force_params():
    external[0] = ti.Vector([0.0,  0.0])
    gravity[0]  = ti.Vector([0.0, -9.8])

@ti.kernel
def build_cloth_model():
    para_init_cloth()
    para_init_restlength()
    

def init_dump_variable():
    pos_new.fill(0)
    sum_num.fill(0)
    pos_old.copy_from(cloth_pos)
    

def update_external_force(fx : ti.f32, fy : ti.f32):
    external[0] = ti.Vector([fx, fy])
    
@ti.func
def fix_cloth_point():
    # Fix Two Point 
    cloth_pos[nparticles - w] = pos_old[nparticles - w]
    cloth_pos[nparticles - 1] = pos_old[nparticles - 1]
    
@ti.kernel
def para_step():
    for i in cloth_pos:
        cloth_vel[i] += (external[0] + gravity[0]) * dt
        cloth_vel[i] *= dumping
        cloth_pos[i] += cloth_vel[i] * dt
    
@ti.func
def para_projection():
    fix_cloth_point()
    for i in cloth_edge:
        v0 = cloth_edge[i][0]
        v1 = cloth_edge[i][1]
        dx = cloth_pos[v0] - cloth_pos[v1]
        scalar = 0.5 * (1.0 - rest_length[i] / dx.norm())
        pos_new[v0] += cloth_pos[v0] - scalar * dx
        pos_new[v1] += cloth_pos[v1] + scalar * dx
        sum_num[v0] += 1
        sum_num[v1] += 1
    
@ti.func
def para_correct():
    for i in cloth_pos:
        pos_tmp[0]    = (pos_new[i] + cloth_pos[i] * alpha) / (sum_num[i] + alpha) 
        cloth_vel[i] += (pos_tmp[0] - cloth_pos[i]) / dt
        cloth_pos[i]  = pos_tmp[0]
    fix_cloth_point()
    
@ti.kernel
def simulator_kernel():
    para_projection()
    para_correct()
        
def init_simulator():
    init_force_params()
    build_cloth_model()

def run_simulator():
    init_dump_variable()
    para_step()
    for k in range(niterator):
        simulator_kernel()
    # debug_arr = external.to_numpy()
    # print(debug_arr)
    
def draw_cloth_mesh_to_screen(pos):
    # scale transform
    pos[:,0] *= padding
    pos[:,1] *= padding
    # translation transform
    pos[:,0] += res_w / 2
    pos[:,1] += res_h / 2
    # projection to screen
    pos[:,0] /= res_w
    pos[:,1] /= res_h
    
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

        draw_cloth_mesh_to_screen(cloth_pos.to_numpy())
        if not isPaused[None]:
            run_simulator()
        gui.show()

if __name__ == '__main__':
    main()