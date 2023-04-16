p_test = cell(4,1);
p_test{1} = [0.1, 0.7; 0.8, 0.3];
p_test{2} = [0.5, 0.1; 0.1, 0.5];
p_test{3} = [0.1, 0.5; 0.5, 0.1];
p_test{4} = [0.9, 0.3; 0.1, 0.3];

marginals = p_test;
n = size(marginals,1);
separators = ones(n-1,2);

for i=1:n-1
    separators(i,:) = sum(marginals{i});
    marginals{i+1} = marginals{i+1}.*(separators(i,:)'*[1,1]);
end

for i=1:n-1
    s_old = separators(n-i,:);
    separators(n-i,:) = sum(marginals{n-i+1},2)';
    marginals{n-i} = marginals{n-i}.*([1;1]*(separators(n-i,:)./s_old));
end

for i=1:n
    marginals{i} = marginals{i}/sum(sum(marginals{i}));
end

disp("(x1,x2)");
disp(marginals{1});
disp("(x2,x3)");
disp(marginals{2});
disp("(x3,x4)");
disp(marginals{3});
disp("(x4,x5)");
disp(marginals{4});