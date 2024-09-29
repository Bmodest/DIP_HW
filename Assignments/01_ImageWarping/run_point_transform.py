import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    #rows=image.shape[0]
    #cols=image.shape[1]
    for k in range(len(target_pts)):
        target_pts[k]=target_pts[k][::-1]
        source_pts[k]=source_pts[k][::-1]

    rows=250
    cols=250

    #warped_image = np.zeros((rows,cols,3))
    warped_image = np.zeros_like(image)
    i_step=max(image.shape[0]//rows,1)
    j_step=max(image.shape[1]//cols,1)
    rows=image.shape[0]//i_step
    cols=image.shape[1]//j_step
    if i_step!=1 :
        rows+=1
    if j_step!=1 :
        cols+=1  
    w_i=np.zeros((rows,cols,len(target_pts)))
    f=np.zeros((rows,cols,2))
    cv_warped_image=warped_image[0:rows,0:cols,:]


    
    v_i=np.minimum(image.shape[0]-1,np.arange(rows)*i_step)
    v_j=np.minimum(image.shape[1]-1,np.arange(cols)*j_step)
    x_grid, y_grid = np.meshgrid(v_i, v_j, indexing='ij')
    x_y_grid = np.stack((x_grid, y_grid),axis=2)

    temp=x_y_grid.reshape(rows, cols, 1, 2) - target_pts
    w_i=1/1/(np.sum(temp ** 2, axis=3)+eps) # [rows, cols, n]

    coeff_sum=np.sum(w_i,axis=2).reshape(rows, cols, 1)
    coeff_sum = np.repeat(coeff_sum, 2, axis=2).reshape(rows, cols, 1, 2) # [rows, cols, 2]

    # [rows, cols, 1, n] * [n, 2] / [rows, cols, 2]
    p_star = (w_i.reshape(rows, cols, 1, len(target_pts)) @  target_pts / coeff_sum).reshape(rows, cols, 2)
    q_star = (w_i.reshape(rows, cols, 1, len(target_pts)) @ source_pts/ coeff_sum).reshape(rows, cols, 2) # [rows, cols, 2]

    #[n, 2] - [rows, cols, 1, 2]
    p_arrow = target_pts - p_star.reshape(rows, cols, 1, 2) #[rows, cols, n, 2]
    q_arrow = source_pts - q_star.reshape(rows, cols, 1, 2) 

    v_sub_p_star=x_y_grid-p_star # [rows, cols, 2]


    k1=v_sub_p_star[...,0]
    k2=v_sub_p_star[...,1]
    k3=np.stack((k1, k2), axis=2) 
    k4=np.stack((k2, -k1), axis=2) 
    k5=np.stack((k3, k4), axis=3).reshape(rows, cols,1,2,2) 
    k5=k5.transpose(0,1,2,4,3)
    a=(np.array([[v_sub_p_star[...,0],v_sub_p_star[...,1]],[v_sub_p_star[...,1],-v_sub_p_star[...,0]]]).reshape(rows, cols, 1,2,2))
    b=(np.array([[p_arrow[...,0],p_arrow[...,1]],[p_arrow[...,1],-p_arrow[...,0]]]).reshape(rows, cols, len(target_pts),2,2))
    c=np.array([[p_arrow[...,0],p_arrow[...,1]],[p_arrow[...,1],-p_arrow[...,0]]])
    d=p_arrow[...,0]
    e=p_arrow[...,1]
    q=np.stack((d, e), axis=3) 
    w=np.stack((e, -d), axis=3) 
    s=np.stack((q,w),axis=4)
    r=s.transpose(0,1,2,4,3)
    c=b@a

    #((np.array([[v_sub_p_star[...,0],v_sub_p_star[...,1]],[v_sub_p_star[...,1],-v_sub_p_star[...,0]]]).reshape(rows, cols,1,2,2)).transpose(0,1,2,4,3))

    q_arrow_times_left_A_i=(q_arrow.reshape(rows, cols, len(target_pts), 1, 2))@(w_i.reshape(rows, cols, len(target_pts),1,1)*s)
    #(rows, cols, len(target_pts), 1, 2)

    f_arrow=q_arrow_times_left_A_i@k5
    f_arrow=np.sum(f_arrow,axis=2).reshape(rows, cols, 2)
    f=np.linalg.norm(v_sub_p_star,axis=2).reshape(rows, cols,1)*f_arrow/((np.linalg.norm(f_arrow,axis=2)+eps).reshape(rows, cols,1))+q_star # [rows, cols, 2]


    valid_x = np.clip(f[:,:,0], 0, image.shape[0]-1).astype(int)
    valid_y = np.clip(f[:,:,1], 0, image.shape[1]-1).astype(int)
    #cv_warped_image=image[int(f[:,:,0]),int(f[:,:,1])]
    cv_warped_image=image[valid_x,valid_y]
    cv_warped_image = cv2.resize(cv_warped_image, (image.shape[1],image.shape[0]), interpolation=cv2.INTER_LINEAR)
    # for i in range(rows):
    #     for j in range(cols):
    #         v_i=min(image.shape[0]-1,i*i_step)
    #         v_j=min(image.shape[1]-1,j*j_step)
    #         temp=[v_i,v_j]-target_pts
    #         # 计算矩阵所有项的平方和
    #         # 计算每一行的平方和
    #         w_i[i,j,:]= 1/(np.sum(temp ** 2, axis=1)+eps)
    #         # for k in range(len(target_pts)):
    #         #     temp=(v_i-target_pts[k][0])**2+(v_j-target_pts[k][1])**2
    #         #     w_i[i,j,k]=1/(temp+eps)
    #         coeff_sum=np.sum(w_i,axis=2)
    #         p_star=np.zeros(2)
    #         q_star=np.zeros(2)
    #         for k in range(len(target_pts)):
    #             p_star=p_star+w_i[i,j,k]*target_pts[k]
    #             q_star=q_star+w_i[i,j,k]*source_pts[k]
    #         p_star=p_star/coeff_sum[i,j]
    #         q_star=q_star/coeff_sum[i,j]
    #         A_i=[]
    #         q_arrow_times_left_A_i=[]
    #         p_arrow=[]
    #         q_arrow=[]
    #         # for i0 in range(i_step):
    #         #     for j0 in range(j_step):
    #         #         if 0<=i*i_step+i0<image.shape[0] and 0<=j*j_step+j0 <image.shape[1] :


    #         #             v_sub_p_star=np.array([i*i_step+i0,j*j_step+j0])-p_star
    #         #             f_arrow=np.array([0,0])
    #         #             for k in range(len(target_pts)):
    #         #                 p_arrow.append(target_pts[k]-p_star)
    #         #                 q_arrow.append(source_pts[k]-q_star)
    #         #                 A_i.append(w_i[i,j,k]*np.array([p_arrow[k],[p_arrow[k][1],-p_arrow[k][0]]])@np.array([v_sub_p_star,[v_sub_p_star[1],-v_sub_p_star[0]]]).T)
    #         #                 f_arrow=f_arrow+q_arrow[k]@A_i[k]
    #         #             f=np.linalg.norm(v_sub_p_star)*f_arrow/np.linalg.norm(f_arrow)+q_star
    #         #             if 0<=f[0]<image.shape[0] and 0<=f[1] <image.shape[1] :
    #         #                 warped_image[i*i_step+i0,j*j_step+j0]=image[int(f[0]),int(f[1])]
    #         v_sub_p_star=np.array([v_i,v_j])-p_star
    #         f_arrow=np.array([0,0])
    #         for k in range(len(target_pts)):
    #             p_arrow.append(target_pts[k]-p_star)
    #             q_arrow.append(source_pts[k]-q_star)
    #             #A_i.append(w_i[i,j,k]*np.array([p_arrow[k],[p_arrow[k][1],-p_arrow[k][0]]])@np.array([v_sub_p_star,[v_sub_p_star[1],-v_sub_p_star[0]]]).T)
    #             q_arrow_times_left_A_i.append(q_arrow[k]@(w_i[i,j,k]*np.array([p_arrow[k],[p_arrow[k][1],-p_arrow[k][0]]])))
    #             #f_arrow=f_arrow+q_arrow[k]@A_i[k]   q_arrow_times_left_A_i[k]@np.array([v_sub_p_star,[v_sub_p_star[1],-v_sub_p_star[0]]]).T
    #             f_arrow=f_arrow+q_arrow_times_left_A_i[k]@np.array([v_sub_p_star,[v_sub_p_star[1],-v_sub_p_star[0]]]).T
    #         f[i,j]=np.linalg.norm(v_sub_p_star)*f_arrow/np.linalg.norm(f_arrow)+q_star
    #         if 0<=f[i,j,0]<image.shape[0] and 0<=f[i,j,1] <image.shape[1] :
    #             warped_image[v_i,v_j]=image[int(f[i,j,0]),int(f[i,j,1])] 
    #             cv_warped_image[i,j]=image[int(f[i,j,0]),int(f[i,j,1])]
    #         # i_start=max(0,(i-1)*i_step)
    #         # j_start=max(0,(j-1)*j_step)
    #         # for i0 in range(v_i-i_start):
    #         #     for j0 in range(v_j-j_start):
    #         #         if i0==0 and j0==0 :
    #         #             continue
    #         #         if i0==0 and j0==v_j-j_start :
    #         #             continue
    #         #         if i0==v_i-i_start and j0==0 :
    #         #             continue
    #         #         if i0==v_i-i_start and j0==v_j-j_start :
    #         #             continue
    #         #         v_sub_p_star=np.array([i_start+i0,j_start+j0])-p_star
    #         #         f_arrow=np.array([0,0])
    #         #         for k in range(len(target_pts)):
    #         #             f_arrow=f_arrow+q_arrow_times_left_A_i[k]@np.array([v_sub_p_star,[v_sub_p_star[1],-v_sub_p_star[0]]]).T
    #         #         f_temp=np.linalg.norm(v_sub_p_star)*f_arrow/np.linalg.norm(f_arrow)+q_star
    #         #         if 0<=f_temp[0]<image.shape[0] and 0<=f_temp[1] <image.shape[1] :
    #         #             warped_image[i_start+i0,j_start+j0]=image[int(f_temp[0]),int(f_temp[1])]
    #         #         else:
    #         #             warped_image[i_start+i0,j_start+j0]=warped_image[i_start,j_start].astype(np.float64)*((v_i-i_start)-i0)*(v_j-j_start-j0)/(v_i-i_start)/(v_j-j_start)+\
    #         #                 warped_image[v_i,j_start].astype(np.float64)*i0*(v_j-j_start-j0)/(v_i-i_start)/(v_j-j_start)+\
    #         #                 warped_image[i_start,v_j].astype(np.float64)*((v_i-i_start)-i0)*j0/(v_i-i_start)/(v_j-j_start)+\
    #         #                 warped_image[v_i,v_j].astype(np.float64)*i0*j0/(v_i-i_start)/(v_j-j_start)
    #         #             #print(i0,' ',j0,' ',warped_image[i_start,j_start],',',warped_image[v_i,j_start],',',warped_image[i_start,v_j],',',warped_image[v_i,v_j],'=',warped_image[i_start+i0,j_start+j0])
    # ##开始插值
    # cv_warped_image = cv2.resize(cv_warped_image, (image.shape[0],image.shape[1]), interpolation=cv2.INTER_LINEAR)

    # # for i in range(rows-1):
    # #     for j in range(cols-1):
    # #         i_end=min((i+1)*i_step,image.shape[0]-1)
    # #         j_end=min((j+1)*j_step,image.shape[1]-1)
    # #         interpolate_warp_img(warped_image=warped_image,i_start=i*i_step,j_start=j*j_step,i_end=i_end,j_end=j_end)
            
    #                     # warped_image[i*i_step+i0,j*j_step+j0]=warped_image[i*i_step,j*j_step].astype(np.float64)*((i_step-i0)/i_step)+warped_image[(i+1)*i_step,j*j_step].astype(np.float64)*(i0/i_step)
    #                     # if  warped_image[(i+1)*i_step,j*j_step,1]==warped_image[i*i_step,j*j_step,1] :
    #                     # print(i0,' ',j0,' ',warped_image[i*i_step,j*j_step],',',warped_image[(i+1)*i_step,j*j_step],',',warped_image[i*i_step,(j+1)*j_step],',',warped_image[(i+1)*i_step,(j+1)*j_step],'=',warped_image[i*i_step+i0,j*j_step+j0],\
    #                     #          warped_image[i*i_step,j*j_step]*(i_step-i0)*(j_step-j0)/i_step/j_step,',',warped_image[(i+1)*i_step,j*j_step].astype(np.float64)*i0)
                  
    ### FILL: 基于MLS or RBF 实现 image warping

    return cv_warped_image

def interpolate_warp_img(warped_image,i_start,j_start,i_end,j_end):
    
    for i0 in range(i_end-i_start):
        for j0 in range(j_end-j_start):
            if i0==0 and j0==0 :
                continue
            if i0==0 and j0==j_end-j_start :
                continue
            if i0==i_end-i_start and j0==0 :
                continue
            if i0==i_end-i_start and j0==j_end-j_start :
                continue
            warped_image[i_start+i0,j_start+j0]=warped_image[i_start,j_start].astype(np.float64)*((i_end-i_start)-i0)*(j_end-j_start-j0)/(i_end-i_start)/(j_end-j_start)+\
                warped_image[i_end,j_start].astype(np.float64)*i0*(j_end-j_start-j0)/(i_end-i_start)/(j_end-j_start)+\
                warped_image[i_start,j_end].astype(np.float64)*((i_end-i_start)-i0)*j0/(i_end-i_start)/(j_end-j_start)+\
                warped_image[i_end,j_end].astype(np.float64)*i0*j0/(i_end-i_start)/(j_end-j_start)
    return warped_image

def interpolate_warp_img2(warped_image,i_start,j_start,i_end,j_end,f):
    for i0 in range(i_end-i_start):
        for j0 in range(j_end-j_start):
            if i0==0 and j0==0 :
                continue
            if i0==0 and j0==j_end-j_start :
                continue
            if i0==i_end-i_start and j0==0 :
                continue
            if i0==i_end-i_start and j0==j_end-j_start :
                continue
            

            warped_image[i_start+i0,j_start+j0]=warped_image[i_start,j_start].astype(np.float64)*((i_end-i_start)-i0)*(j_end-j_start-j0)/(i_end-i_start)/(j_end-j_start)+\
                warped_image[i_end,j_start].astype(np.float64)*i0*(j_end-j_start-j0)/(i_end-i_start)/(j_end-j_start)+\
                warped_image[i_start,j_end].astype(np.float64)*((i_end-i_start)-i0)*j0/(i_end-i_start)/(j_end-j_start)+\
                warped_image[i_end,j_end].astype(np.float64)*i0*j0/(i_end-i_start)/(j_end-j_start)
    return warped_image


def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
