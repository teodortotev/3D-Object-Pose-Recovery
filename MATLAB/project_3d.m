%% Project the 3D points to generate 2D points according to the viewpoint
function x = project_3d(cads, object)

% Get 3D vertice coordinates
vert = cads.vertices(:,1:3);
fac = cads.faces;

%% Get viewpoint properties
if isfield(object, 'viewpoint') == 1
    viewpoint = object.viewpoint;               % Get the viewpoint
    a = viewpoint.azimuth*pi/180;               % Azimuth (0-360) in radians (phi)
    e = viewpoint.elevation*pi/180;             % Elevation (0-180) in radians (theta)
    d = viewpoint.distance;                     % Distance to camera
    f = viewpoint.focal;                        % Focal length of the camera                        
    theta = viewpoint.theta*pi/180;             % ?????
    principal = [viewpoint.px viewpoint.py];    % Camera principal point
    viewport = viewpoint.viewport;              % ?????
else
    x = [];
    return;
end

if d == 0
    x = [];
    return;
end

%% Find camera center coordinates
C = zeros(3,1);
C(1) = d*cos(e)*sin(a);
C(2) = -d*cos(e)*cos(a);
C(3) = d*sin(e);

%% Rotate coordinate system by theta is equal to rotating the model by -theta.
a = -a;
e = -(pi/2-e);

%% Define rotation matrix
Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];   % Rotate by a
Rx = [1 0 0; 0 cos(e) -sin(e); 0 sin(e) cos(e)];   % Rotate by e
R = Rx*Rz;                                         % Stack in reverse

%% Perspective projection matrix
% however, we set the viewport to 3000, which makes the camera similar to
% an affine-camera. Exploring a real perspective camera can be a future work.
M = viewport;
P = [M*f 0 0; 0 M*f 0; 0 0 -1] * [R -R*C];

%% Get depth of each point
for i = 1:length(vert)
    depth(i) = sqrt((C(1)-vert(i, 1))^2 + (C(2)-vert(i, 2))^2 + (C(3)-vert(i, 3))^2);
end

%% Project 3d points to 2d
x = P*[vert ones(size(vert,1), 1)]';
x(1,:) = x(1,:) ./ x(3,:);
x(2,:) = x(2,:) ./ x(3,:);
x = x(1:2,:);

% rotation matrix 2D
R2d = [cos(theta) -sin(theta); sin(theta) cos(theta)];
x = (R2d * x)';
% x = x';

% Transform to image coordinates and add depth
x(:,2) = -1 * x(:,2);
x = x + repmat(principal, size(x,1), 1);
x(:,3) = depth';