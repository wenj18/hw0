function f=alt(a,k)
b=a;
for i=1:k
    a(i)=a(length(a)+i-k);
end
for i=k+1:length(a)
    a(i)=b(i-k);
end
f=a;