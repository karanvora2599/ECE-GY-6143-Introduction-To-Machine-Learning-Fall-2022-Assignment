Training = [0.8,0.2;0.2,0.8];
Estimation = [0.4,0.1,0.3,0.2;0.1,0.4,0.2,0.3];
Observation = [1,4,2,2,3];
InitialState = [1,0];
Var = size(Training, 1);

Number = size(Observation, 2);
p1 = zeros(Var, Var, Number);
p2 = zeros(Var, Number);
p2(:, 1) = InitialState;

for i = 2 : Number
    pres = Observation(1, i);
    p1(:, :, i) = diag(p2(:, i - 1)) * Training * diag(Estimation(:,pres));
    p2(:, i) = max(p1(:, :, i));
end

for i = Number - 1 : -1 : 1
    p3 = max(p1(:, :, i + 1), [], 2);
    p1(:, :, i) = p1(:, :, i) * diag(p3 ./ p2(:, i));
    p2(:, i) = p3;
end

[A,H] = max(p2);
disp("final emission probabilities")
disp(p2)
disp("state")
disp(H)