close all
train = [-122.542
    -110.314
    -101.0720
    -95.437
 ]
test =[
    -131.649
    -116.0916
    -105.3082
    -98.528
    ]

N = [10
    100
    1000
    10000
    ]
semilogx(N,train)
hold on
semilogx(N,test)
legend("Average LL : train", "Average LL: test")
xlabel("Number of Sequences")
ylabel("Log-Likelihood")