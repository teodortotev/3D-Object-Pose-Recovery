function opt = globals()

path_pascal3d = '../../Data'; 
opt.path_pascal3d = path_pascal3d;
opt.path_img_pascal = [opt.path_pascal3d '/Images/%s_pascal'];
opt.path_ann_pascal = [opt.path_pascal3d '/Annotations/%s_pascal'];
opt.path_img_imagenet = [opt.path_pascal3d '/Images/%s_imagenet'];
opt.path_ann_imagenet = [opt.path_pascal3d '/Annotations/%s_imagenet'];
opt.path_cad = [opt.path_pascal3d '/CAD/%s.mat'];