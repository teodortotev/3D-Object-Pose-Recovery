% Show Pascal3D+ Annotations
function main

cls = 'car'; % Specify the class
opt = globals();

colors = load(fullfile(opt.path_pascal3d, '/CAD/colors.mat'));
colors = colors.colors;

%% Load PASCAL3D+ Modified CAD Models
fprintf('Load CAD Models from File\n');
file = fullfile(opt.path_pascal3d, '/CAD/labelled_cads_centre.mat');
file = load(file);
cads = file.cads;

%% Load Annotations
filename = fullfile(sprintf(opt.path_ann_pascal, cls), '*.mat');
files = dir(filename);

%% Get Number of Images
nimages = length(files);

%% Loop through the images
bin_corrects.total = 0
bin_corrects.four = 0;
bin_corrects.eight = 0;
bin_corrects.steen = 0;
bin_corrects.tfour = 0;
for img_idx = 34:nimages
    filename = files(img_idx).name
    [pathstr, name, ext] = fileparts(filename);
    
    fprintf('%d %s\n', img_idx, filename);
    im_name = filename;
    
    %% Load Image
    filename = fullfile(sprintf(opt.path_img_pascal, cls), [name '.jpg']);
    I = imread(filename);
    [h, w, ~] = size(I);
    
    %% Load Annotations
    filename = fullfile(sprintf(opt.path_ann_pascal, cls), files(img_idx).name);
    object = load(filename);
    objects = object.record.objects;
    
    %% For All Annotated Objects Do
    for i = 1:numel(objects)
        atleastone = 0;
        object = objects(i);
        if strcmp(object.class, cls) == 0
            %disp('Classes do not match!')
            continue;
        end
        
        bin_corrects.total = bin_corrects.total + 1;
        % Find mean part positions 2D
        mask_file = fullfile(opt.path_pascal3d, '/Masks/car_pascal/single', strcat(im_name(1:end-4),'_mask_', int2str(i), '.mat'));
        if isfile(mask_file)
            single_mask = load(fullfile(opt.path_pascal3d, '/Masks/car_pascal/single', strcat(im_name(1:end-4),'_mask_', int2str(i), '.mat'))).single_mask;
        else
            continue;
        end
        x2d_mask.count = [];
        x2d_mask.x = [];
        for part = 1:8
            [row, col] = find(single_mask == part);
            count = length(row);
            x2d_mask.count = [x2d_mask.count; count];
            %x2d_mask.x = [x2d_mask.x; sum(row)/count sum(col)/count];
            if count ~= 0
                x2d_mask.x = [x2d_mask.x; (max(row)+min(row))/2 (max(col)+min(col))/2];
            else
                x2d_mask.x = [x2d_mask.x; NaN NaN];
            end
        end
        
        %% Create mask image for vizualization
%         pic = zeros(h,w,3);
%         for variety = 1:8
%            for height = 1:h
%                for width = 1:w
%                    if single_mask(height,width) == variety
%                        pic(height, width, :) = colors(variety, :);
%                    end
%                end
%            end
%         end
%       imshow(pic)
%         axis on;
%         hold on;
%         set(gca, 'Visible', 'off')
        [out,indx] = sort(x2d_mask.count);
%         plot(x2d_mask.x(indx(end-3:end),2), x2d_mask.x(indx(end-3:end),1), 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'red');
        
        % Find mean part positions 3D
        cad_index = object.cad_index;
        vertices = cads(cad_index).vertices(:,1:3);
        clas = cads(cad_index).vertices(:,4);
        x3d_mask.count = [];
        x3d_mask.x = [];
        for part = 1:8
            idx = find(clas == part);
            count = length(idx);
            x3d_mask.count = [x3d_mask.count; count];
            x3d_mask.x = [x3d_mask.x; sum(vertices(idx,1))/count sum(vertices(idx,2))/count sum(vertices(idx,3))/count ];
            %x3d_mask.x = [x3d_mask.x; (max(vertices(idx,1))+min(vertices(idx,1)))/2 (max(vertices(idx,2))+min(vertices(idx,2)))/2 (max(vertices(idx,3))+min(vertices(idx,3)))/2 ];
        end
        
        % Find non-empty        
        viewpoint = object.viewpoint;
        if sum(x2d_mask.count ~=0) < 4
           continue;
        end
        
        v_out = init(viewpoint, x2d_mask.x(indx(end-3:end),:), x3d_mask.x(indx(end-3:end),:), h, w, indx(end-3:end));
        
        if (viewpoint.azimuth - 45) <= v_out(1) && v_out(1) <= (viewpoint.azimuth + 45)
           bin_corrects.four = bin_corrects.four + 1;
           if viewpoint.azimuth - 22.5 <= v_out(1) && v_out(1) <= viewpoint.azimuth + 22.5
              bin_corrects.eight = bin_corrects.eight + 1;
              if viewpoint.azimuth - 11.25 <= v_out(1) && v_out(1) <= viewpoint.azimuth + 11.25
                 bin_corrects.steen = bin_corrects.steen + 1;
                 if viewpoint.azimuth - 7.5 <= v_out(1) && v_out(1) <= viewpoint.azimuth + 7.5
                     bin_corrects.tfour = bin_corrects.tfour + 1;
                 end
              end
           end
        end        
%         bin_corrects
        % Display 3D bounding box + IoU
        iou = box3d(viewpoint, v_out, cads(cad_index));
%         alpha 0.3;
%         scatter3(x3d_mask.x(indx(end-3:end),1), x3d_mask.x(indx(end-3:end),2), x3d_mask.x(indx(end-3:end),3), 500, 'red', 'filled');
    end
end
end

function v_out = init(viewpoint, x2d, x3d, h, w, indx)
% inialization
v0 = zeros(7,1);
% azimuth
%a = viewpoint.azimuth_coarse;
r = -1 + 2*rand(1,1);
a = 180 + r*180;
v0(1) = a*pi/180;
margin = 180;
%margin = 25;
aextent = [max(a-margin,0)*pi/180 min(a+margin,360)*pi/180];
% elevation
%e = viewpoint.elevation_coarse;
r = -1 + 2*rand(1,1);
e = 0 + r*90;
v0(2) = e*pi/180;
margin = 90;
%margin = 15;
eextent = [max(e-margin,-90)*pi/180 min(e+margin,90)*pi/180];
% distance
dextent = [0, 100];
v0(3) = compute_distance(v0(1), v0(2), dextent, x2d, x3d);
d = v0(3);
%v0(3) = viewpoint.distance;
%d = viewpoint.distance;
margin = 25;
dextent = [max(d-margin,0) min(d+margin,100)];
% focal length
v0(4) = 1;
fextent = [1 1];
% principal point
[principal_point, lbp, ubp] = compute_principal_point(v0(1), v0(2), v0(3), x2d, x3d);
v0(5) = principal_point(1);
v0(6) = principal_point(2);
%v0(5) = h/2;
%lbp = [0 0];
%v0(6) = w/2;
%ubp = [h w];
% in-plane rotation
r = -1 + 2*rand(1,1);
v0(7) = r*pi;
rextent = [-pi, pi];
% lower bound
lb = [aextent(1); eextent(1); dextent(1); fextent(1); lbp(1); lbp(2); rextent(1)];
% upper bound
ub = [aextent(2); eextent(2); dextent(2); fextent(2); ubp(1); ubp(2); rextent(2)];

% optimization
v_out = zeros(10,1);
[v_out(1), v_out(2), v_out(3), v_out(4), v_out(5), v_out(6), v_out(7), v_out(8), v_out(9), v_out(10)]...
    = compute_viewpoint_one(v0, lb, ub, x2d, x3d);
end

% compute the initial distance
function distance = compute_distance(azimuth, elevation, dextent, x2d, x3d)

% compute pairwise distance
n = size(x2d, 1);
num = n*(n-1)/2;
d2 = zeros(num,1);
count = 1;
for i = 1:n
    for j = i+1:n
        d2(count) = norm(x2d(i,:)-x2d(j,:));
        count = count + 1;
    end
end

% optimization
options = optimset('Algorithm', 'interior-point', 'Display', 'none');
distance = fmincon(@(d)compute_error_distance(d, azimuth, elevation, d2, x3d),...
    (dextent(1)+dextent(2))/2, [], [], [], [], dextent(1), dextent(2), [], options);

end

function error = compute_error_distance(distance, azimuth, elevation, d2, x3d)

a = azimuth;
e = elevation;
d = distance;
f = 1;

% camera center
C = zeros(3,1);
C(1) = d*cos(e)*sin(a);
C(2) = -d*cos(e)*cos(a);
C(3) = d*sin(e);

a = -a;
e = -(pi/2-e);

% rotation matrix
Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];   %rotate by a
Rx = [1 0 0; 0 cos(e) -sin(e); 0 sin(e) cos(e)];   %rotate by e
R = Rx*Rz;

% perspective project matrix
M = 3000;
P = [M*f 0 0; 0 -M*f 0; 0 0 -1] * [R -R*C];

% project
x = P*[x3d ones(size(x3d,1), 1)]';
x(1,:) = x(1,:) ./ x(3,:);
x(2,:) = x(2,:) ./ x(3,:);
x = x(1:2,:)';

% compute pairwise distance
n = size(x, 1);
num = n*(n-1)/2;
d3 = zeros(num,1);
count = 1;
for i = 1:n
    for j = i+1:n
        d3(count) = norm(x(i,:)-x(j,:));
        count = count + 1;
    end
end

% compute error
error = norm(d2-d3);

end

function [center, lb, ub] = compute_principal_point(a, e, d, x2d, x3d)

f = 1;
% camera center
C = zeros(3,1);
C(1) = d*cos(e)*sin(a);
C(2) = -d*cos(e)*cos(a);
C(3) = d*sin(e);

a = -a;
e = -(pi/2-e);

% rotation matrix
Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];   %rotate by a
Rx = [1 0 0; 0 cos(e) -sin(e); 0 sin(e) cos(e)];   %rotate by e
R = Rx*Rz;

% perspective project matrix
M = 3000;
P = [M*f 0 0; 0 -M*f 0; 0 0 -1] * [R -R*C];

% project
x = P*[x3d ones(size(x3d,1), 1)]';
x(1,:) = x(1,:) ./ x(3,:);
x(2,:) = x(2,:) ./ x(3,:);
x = x(1:2,:)';

% project object center
c = P*[0 0 0 1]';
c = c ./ c(3);
c = c(1:2)';

% predict object center
cx2 = c(1);
cy2 = c(2);
center = [0 0];
for i = 1:size(x2d,1)
    cx1 = x(i,1);
    cy1 = x(i,2);
    dc = sqrt((cx1-cx2)*(cx1-cx2) + (cy1-cy2)*(cy1-cy2));
    ac = atan2(cy2-cy1, cx2-cx1);
    center(1) = center(1) + x2d(i,1) + dc*cos(ac);
    center(2) = center(2) + x2d(i,2) + dc*sin(ac);
end
center = center ./ size(x2d,1);

width = 0;
height = 0;
for i = 1:size(x2d,1)
    w = abs(x2d(i,1)-center(1));
    if width < w
        width = w;
    end
    h = abs(x2d(i,2)-center(2));
    if height < h
        height = h;
    end
end

% lower bound and upper bound
lb = [center(1)-width/10 center(2)-height/10];
ub = [center(1)+width/10 center(2)+height/10];

end

% compute viewpoint angle from 2D-3D correspondences
function [azimuth, elevation, distance, focal, px, py, theta, error, interval_azimuth, interval_elevation]...
    = compute_viewpoint_one(v0, lb, ub, x2d, x3d)

options = optimset('Algorithm', 'interior-point', 'Display', 'none');
[vp, fval] = fmincon(@(v)compute_error(v, x2d, x3d),...
    v0, [], [], [], [], lb, ub, [], options);

viewpoint = vp;
error = fval;

azimuth = viewpoint(1)*180/pi;
if azimuth < 0
    azimuth = azimuth + 360;
end
if azimuth >= 360
    azimuth = azimuth - 360;
end
elevation = viewpoint(2)*180/pi;
distance = viewpoint(3);
focal = viewpoint(4);
px = viewpoint(5);
py = viewpoint(6);
theta = viewpoint(7)*180/pi;

% estimate confidence inteval
v = viewpoint;
v(7) = 0;
x = project(v, x3d);
% azimuth
v = viewpoint;
v(1) = v(1) + pi/180;
xprim = project(v, x3d);
error_azimuth = sum(diag((x-xprim) * (x-xprim)'));
interval_azimuth = error / error_azimuth;
% elevation
v = viewpoint;
v(2) = v(2) + pi/180;
xprim = project(v, x3d);
error_elevation = sum(diag((x-xprim) * (x-xprim)'));
interval_elevation = error / error_elevation;

end

function error = compute_error(v, x2d, x3d)

a = v(1);
e = v(2);
d = v(3);
f = v(4);
principal_point = [v(5) v(6)];
theta = v(7);

% camera center
C = zeros(3,1);
C(1) = d*cos(e)*sin(a);
C(2) = -d*cos(e)*cos(a);
C(3) = d*sin(e);

a = -a;
e = -(pi/2-e);

% rotation matrix
Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];   %rotate by a
Rx = [1 0 0; 0 cos(e) -sin(e); 0 sin(e) cos(e)];   %rotate by e
R = Rx*Rz;

% perspective project matrix
M = 3000;
P = [M*f 0 0; 0 M*f 0; 0 0 -1] * [R -R*C];

% project
x = P*[x3d ones(size(x3d,1), 1)]';
x(1,:) = x(1,:) ./ x(3,:);
x(2,:) = x(2,:) ./ x(3,:);
x = x(1:2,:);

% rotation matrix 2D
R2d = [cos(theta) -sin(theta); sin(theta) cos(theta)];
x = (R2d * x)';
% compute error
error = normal_dist(x, x2d, principal_point);

end

% re-projection error
function error = normal_dist(x, x2d, p_pnt)

error = 0;
for i = 1:size(x2d, 1)
    point = x2d(i,:) - p_pnt;
    point(2) = -1 * point(2);
    error = error + (point-x(i,:))*(point-x(i,:))'/size(x2d, 1);
end
end

function x = project(v, x3d)

a = v(1);
e = v(2);
d = v(3);
f = v(4);
theta = v(7);

% camera center
C = zeros(3,1);
C(1) = d*cos(e)*sin(a);
C(2) = -d*cos(e)*cos(a);
C(3) = d*sin(e);

a = -a;
e = -(pi/2-e);

% rotation matrix
Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];   %rotate by a
Rx = [1 0 0; 0 cos(e) -sin(e); 0 sin(e) cos(e)];   %rotate by e
R = Rx*Rz;

% perspective project matrix
M = 3000;
P = [M*f 0 0; 0 M*f 0; 0 0 -1] * [R -R*C];

% project
x = P*[x3d ones(size(x3d,1), 1)]';
x(1,:) = x(1,:) ./ x(3,:);
x(2,:) = x(2,:) ./ x(3,:);
x = x(1:2,:);

% rotation matrix 2D
R2d = [cos(theta) -sin(theta); sin(theta) cos(theta)];
x = (R2d * x)';

end

% project the CAD model to generate aspect part locations
function x = project_3d_points(x3d, object)

if isfield(object, 'viewpoint') == 1
    % project the 3D points
    viewpoint = object.viewpoint;
    a = viewpoint.azimuth*pi/180;
    e = viewpoint.elevation*pi/180;
    d = viewpoint.distance;
    f = viewpoint.focal;
    theta = viewpoint.theta*pi/180;
    principal = [viewpoint.px viewpoint.py];
    viewport = viewpoint.viewport;
else
    x = [];
    return;
end

if d == 0
    x = [];
    return;
end

% camera center
C = zeros(3,1);
C(1) = d*cos(e)*sin(a);
C(2) = -d*cos(e)*cos(a);
C(3) = d*sin(e);

a = -a;
e = -(pi/2-e);

% rotation matrix
Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];   %rotate by a
Rx = [1 0 0; 0 cos(e) -sin(e); 0 sin(e) cos(e)];   %rotate by e
R = Rx*Rz;

% perspective project matrix
M = viewport;
P = [M*f 0 0; 0 M*f 0; 0 0 -1] * [R -R*C];

% project
x = P*[x3d ones(size(x3d,1), 1)]';
x(1,:) = x(1,:) ./ x(3,:);
x(2,:) = x(2,:) ./ x(3,:);
x = x(1:2,:);

% rotation matrix 2D
R2d = [cos(theta) -sin(theta); sin(theta) cos(theta)];
x = (R2d * x)';
% x = x';

% transform to image coordinates
x(:,2) = -1 * x(:,2);
x = x + repmat(principal, size(x,1), 1);

end

function iou=box3d(viewpoint, pred_viewpoint, model)
trimesh(model.faces, model.vertices(:,1), model.vertices(:,2), model.vertices(:,3), 'EdgeColor', 'b');
axis on;
hold on;
axis equal;
set(gca, 'Visible', 'off')
iou=1

% Define basic cube boundaries with volume 1
m.x = [min(model.vertices(:,1)), max(model.vertices(:,1))];
m.y = [min(model.vertices(:,2)), max(model.vertices(:,2))];
m.z = [min(model.vertices(:,3)), max(model.vertices(:,3))];

%Volume = (m.x(2)-m.x(1))*(m.y(2)-m.y(1))*(m.z(2)-m.z(1));
%m.x = m.x*nthroot(1/Volume,3);
%m.y = m.y*nthroot(1/Volume,3);
%m.z = m.z*nthroot(1/Volume,3);

% Create cubes
cube = createcube(m.x,m.y,m.z);
shp1 = alphaShape(cube(:,1), cube(:,2), cube(:,3));
% plot(shp1)
% hold on;


% Rotate the cube appropriately
cube = applyviewpoint(cube,viewpoint);
%shp2 = alphaShape(cube(:,1), cube(:,2), cube(:,3));
shp2 = alphaShape(cube(:,1), cube(:,2));
% plot(shp2)
%corners = [m.x(1), m.y(1), m.z(1);...
%           m.x(1), m.y(1), m.z(2);...
%          m.x(1), m.y(2), m.z(1);...
%          m.x(1), m.y(2), m.z(2);...
%           m.x(2), m.y(1), m.z(1);...
%           m.x(2), m.y(1), m.z(2);...
%           m.x(2), m.y(2), m.z(1);...
%           m.x(2), m.y(2), m.z(2);];
       
%scatter3(corners(:,1), corners(:,2), corners(:,3), 100, 'red', 'filled');
%x1 = cube(:,1);
%y1 = cube(:,2);
%z1 = cube(:,3);

%x2 = cube(:,1) + 0.1;
%y2 = cube(:,2) + 0.1;
%z2 = cube(:,3) + 0.1;

%shp1 = alphaShape(x1,y1,z1);
%volume(shp1)
%shp2 = alphaShape(x2,y2,z2);
%volume(shp2)
%plot(shp1)
%hold on;
%plot(shp2)
%hold on

%id1=inShape(shp2,x1,y1,z1);
%id2=inShape(shp1,x2,y2,z2);
%shp3=alphaShape([x1(id1); x2(id2)], [y1(id1); y2(id2)], [z1(id1); z2(id2)]);
%volume(shp3)
%plot(shp3)
end

function h=createcube(mx,my,mz)
%# these don't all have to be the same
x = mx(1):0.05:mx(2); y = my(1):0.05:my(2); z = mz(1):0.05:mz(2);

[X1 Y1 Z1] = meshgrid(x([1 end]),y,z);
X1 = permute(X1,[2 1 3]); Y1 = permute(Y1,[2 1 3]); Z1 = permute(Z1,[2 1 3]);
%X1(end+1,:,:) = NaN; Y1(end+1,:,:) = NaN; Z1(end+1,:,:) = NaN;
[X2 Y2 Z2] = meshgrid(x,y([1 end]),z);
%X2(end+1,:,:) = NaN; Y2(end+1,:,:) = NaN; Z2(end+1,:,:) = NaN;
[X3 Y3 Z3] = meshgrid(x,y,z([1 end]));
X3 = permute(X3,[3 1 2]); Y3 = permute(Y3,[3 1 2]); Z3 = permute(Z3,[3 1 2]);
%X3(end+1,:,:) = NaN; Y3(end+1,:,:) = NaN; Z3(end+1,:,:) = NaN;

%#figure('Renderer','opengl')
%h = line([X1(:);X2(:);X3(:)], [Y1(:);Y2(:);Y3(:)], [Z1(:);Z2(:);Z3(:)]);
h = [[X1(:);X2(:);X3(:)], [Y1(:);Y2(:);Y3(:)], [Z1(:);Z2(:);Z3(:)]];
%set(h, 'Color',[0.5 0.5 1], 'LineWidth',1, 'LineStyle','-')

%#set(gca, 'Box','on', 'LineWidth',2, 'XTick',x, 'YTick',y, 'ZTick',z, ...
%#  'XLim',[x(1) x(end)], 'YLim',[y(1) y(end)], 'ZLim',[z(1) z(end)])
%#xlabel x, ylabel y, zlabel z
%axis off
%view(3), axis vis3d
%camproj perspective, rotate3d on
end

function h=applyviewpoint(cube, viewpoint)
a = viewpoint.azimuth;
e = viewpoint.elevation;
d = viewpoint.distance;
f = viewpoint.focal;
theta = viewpoint.theta;

% camera center
C = zeros(3,1);
C(1) = d*cos(e)*sin(a);
C(2) = -d*cos(e)*cos(a);
C(3) = d*sin(e);

a = -a;
e = -(pi/2-e);

% rotation matrix
Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];   %rotate by a
Rx = [1 0 0; 0 cos(e) -sin(e); 0 sin(e) cos(e)];   %rotate by e
R = Rx*Rz;

% perspective project matrix
M = 3000;
P = [M*f 0 0; 0 -M*f 0; 0 0 -1] * [R -R*C];
%P = [R -R*C];

% project
h = P*[cube ones(size(cube,1), 1)]';
h(1,:) = h(1,:) ./ h(3,:);
h(2,:) = h(2,:) ./ h(3,:);
h = h(1:2,:);

R2d = [cos(theta) -sin(theta); sin(theta) cos(theta)];
h = (R2d * h)';
%h = h';
end