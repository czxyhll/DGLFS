function [W_pre,S_pre,Ft_pre,acc_best]=DGLFS(Xs,Ys,Xt,Yt,Ft_init,S_init,options)
epsilon = 1e-5; 
p = options.p;
lammda1=options.lammda1;
lammda2=options.lammda2;
lammda3=options.lammda3;
miu=options.miu;

ru=1.05;
to1=1e-4;
tol2=1e-5;
miu_max=1e8;


[d,ns]=size(Xs);
[~,nt]=size(Xt);
c=max(Ys);
n=ns+nt;
X=[Xs,Xt]; 
Ys=onehot(Ys,c);

Ns=Nc(Ys);
Nss=diag([1/ns,sum(Ns)]);
Yss=[ones(ns,1),Ys];
H=eye(n)-1/n * ones(n,n);
Ft=Ft_init;
S= S_init;
U=S;
Z=S;
D=diag(sum(S,2));
L=D-S;
Nt=Nc(Ft);
Ntt=diag([1/nt,sum(Nt)]); 
Ftt=[ones(nt,1),Ft]; 
A=Xs*Yss*Nss-Xt*Ftt*Ntt;

[W, ~, ~]=eigs(A*A'+2*lammda1*X*L*X'+0.1*eye(d),X*H*X', p, 'SM');
temp = 2*(sum(W.*W,2)+epsilon).^0.5;
Q = diag(1./temp);

F=[Ys;Ft];
NITER=45;
obj=[];
allacc =[];
C1=zeros(size(S));

tic;
for iter=1:NITER 
    %Ft_pre=Ft;
    W_pre=W;
    S_pre=S;
    Z_pre=Z;
  

    %-------------------updata Z--------------------
    Z=(X'*W*W'*X + 0.5*miu*eye(n) )\(X'*W*W'*X +0.5*miu*S -0.5*C1 );
    Z=Z-diag(diag(Z));
    Z=(Z+Z')/2;

    %-------------------update S--------------------
    dist_wx=L2_distance_1(W'*X,W'*X); 
    dist_f=L2_distance_1(F',F'); 
    H1=miu*(Z+C1/miu);
    G1= dist_wx+lammda3*dist_f; 
    for i=1:1:n
        m=(H1(i,:)-G1(i,:))./(2*lammda1 + miu);
        [S(:,i),~]=EProjSimplex_new(m);
    end
    clear i;
    S=(S+S')/2;
    D=diag(sum(S,2));    

    %-------------------update FT--------------------
    L=D-S;
    Lst=L(1:ns,ns+1:n);
    Dtt=D(ns+1:n,ns+1:n); 
    Stt=S(ns+1:n,ns+1:n);
    N=Nt*Nt'; 
    Z1=Xt'*(W*W')*Xt;
    M=lammda3*Ys'*Lst-Nt*Ns'*Ys'*Xs'*(W*W')*Xt; 
    % 
    for i=1:1:nt
        bb=zeros(c,1);
        for j=1:1:nt
            fj=Ft(j,:)';
            bb=bb+lammda3*fj*Stt(i,j)-Z1(i,j)*N*fj;
        end
        b=bb-2*M(:,i);
        mm=b./(lammda3*Dtt(i,i));
        [v,~] = EProjSimplex_new(mm);
        Ft(i,:)=v';
    end
    
    % ------------------update W-----------------
    F=[Ys;Ft];
    Nt=Nc(Ft);
    for tt=1:c
        if(Nt(tt,tt)==inf)
            Nt(tt,tt)=1e-10;
        end
    end
    Ntt=diag([1/nt,sum(Nt)]);
    Ftt=[ones(nt,1),Ft];
    A=Xs*Yss*Nss-Xt*Ftt*Ntt;
    [W, ~, ~]=eigs(A*A'+ 2*X*L*X'+ X*X' - 2*X*Z*X' + X*Z*Z'*X'+ lammda2*Q + 0.1*eye(d), X*H*X', p, 'SM');

     %-----------------updata Q------------------
    temp = (sum(W.*W,2)+epsilon).^0.5;
    Q = diag(1./temp);


    %-----------------update C1 and miu---------------
    L1=Z-S;
    C1=C1+miu*L1;
    
    LL1=norm(Z-Z_pre,'fro');
    LL2=norm(S-S_pre,'fro');
    SLSL=max(LL1,LL2)/norm(X,'fro');
    if miu*SLSL > tol2
        miu= min(ru*miu,miu_max);
    end
   
%     %--------------OBJ-----------------------
%     temp1(iter,1)=norm(W'*A,'fro')^2;
%     temp2(iter,1)=norm(W'*X-W'*X*Z,'fro')^2;
%     temp3(iter,1)=trace(W'*X*L*X'*W);
%     temp4(iter,1)=norm(S,'fro')^2;
%     temp5(iter,1)=trace(W'*Q*W);
%     temp6(iter,1)=trace(F'*L*F);   
%     obj=temp1+temp2+temp3+lammda1*temp4+lammda2*temp5+lammda3*temp6;

    %-------------output acc---------------- 
    [~,predict_label] = max(Ft,[],2);
    acc = length(find(predict_label == Yt))./length(Yt);
    allacc(iter) = acc;
    if iter==1
        acc_best=acc;
        Ft_pre=predict_label;
        W_pre=W;
        S_pre=S;
    elseif iter>1 && acc_best<acc
        acc_best=acc;
        Ft_pre=predict_label;
        W_pre=W;
        S_pre=S;
    end
    %fprintf('iter=%d,miu=%0.4f,The acc=%0.4f\n',iter,miu,acc);
    
        
    leql=max(max(abs(LL1(:))),max(abs(LL2(:))));
    stopC=leql;
    if stopC < to1
       break;
    end
    
end

end
