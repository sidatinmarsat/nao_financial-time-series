import numpy as np
from sklearn import preprocessing
from datetime import datetime
import sys

def readentry(line):
    temp = line.rstrip().split(",")
    return float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4])

def gettime(line):
    temp = [int(i) for i in line.split(" ")[0].split("-")+line.split(" ")[1].split(":")][:-1]
    return datetime(*temp)

def readduka(filename, resolution=60):
    raw = np.array([i.rstrip().split(",") for i in open(filename, "r").readlines()])
    data = raw[:,1:].astype(float)
    
    # base time:
    base = datetime(2000, 1, 1)
    temp = [gettime(i)-base for i in raw[:, 0]]
    time = np.array([i.days*24*resolution+i.seconds/60/(60/resolution) for i in temp])
    day = np.array([i.seconds/60/60/4 for i in temp])
    return data, time, day, raw[:, 0]

def differ(data, x=1):
    for j in range(x):
        ret = np.zeros((data.shape[0]-1, data.shape[1]))
        for i in range(data.shape[0]-1):
            ret[i] = data[i+1]-data[i]
        data = ret
    return data
test = np.arange(12).reshape((3,4))
assert np.sum(differ(test)) == 32

def get_uncommon_index_from_ordered_list_of_lists(listoflists, debug=0):
    lol = listoflists
    
    # this potentiall takes a lot of time
    curset = set(lol[0])
    for i in lol[1:]:
        curset = curset.intersection(i)
        
    common = sorted(list(curset))
    ret = []
    for l in listoflists:
        index = 0
        skip = []
        i = 0
        for i in range(len(l)):
            if index < len(common):
                if l[i] < common[index]:
                    skip.append(i)
                else:
                    while index < len(common) and common[index] <= l[i]:
                        index += 1
            else:
                skip.append(i)
        ret.append(skip)
        print len(common)
        print len(l)
        print len(skip)
        assert len(skip) == len(l)-len(common)
    return ret

def process_data(filenames, fxnames, resolution=60, suffix="test"):
    ## read in data
    data = {}
    times = {}
    hours = {}
    raws = {}
    count = 0
    for filename in filenames:
        tempdata, time, hour, raw = readduka(filename, resolution)
        
        # Invert if USD comes first
        if fxnames[count][:3] == "USD": 
            tempdata = 1.0/tempdata
            temp = tempdata[:, 2]
            tempdata[:, 2] = tempdata[:, 3]
            tempdata[:, 3] = temp
        
        data[fxnames[count]] = tempdata
        times[fxnames[count]] = time
        hours[fxnames[count]] = hour
        raws[fxnames[count]] = raw
        count+=1
        
    ## remove entries that are empty
    removethese = get_uncommon_index_from_ordered_list_of_lists([times[fn] for fn in fxnames])
    count = 0
    for i in removethese:
        print "removing", len(i)
        count +=1
    np.save("removethese."+suffix, removethese)
    prev = set()
    for i in range(len(fxnames)):
        mask = np.ones(data[fxnames[i]].shape[0]).astype(np.bool_)
        mask[removethese[i]] = False
        print len(prev.intersection(set(times[fxnames[i]])))
        prev = set(times[fxnames[i]])
        print fxnames[i], data[fxnames[i]].shape
        data[fxnames[i]] = data[fxnames[i]][mask]
        times[fxnames[i]] = times[fxnames[i]][mask]
        hours[fxnames[i]] = hours[fxnames[i]][mask]
    
    #returns (t+1)-t
    returns = np.zeros((data[fxnames[0]].shape[0]-1, len(fxnames)))
    for i in range(len(fxnames)):
        print len(data[fxnames[i]])
        print data[fxnames[i]].shape
        print returns.shape
        print differ(data[fxnames[i]][:, :1])[:,0].shape
        print data[fxnames[i]][:-1,0].shape
        print
        returns[:, i] = np.divide(differ(data[fxnames[i]][:, :1])[:,0], data[fxnames[i]][:-1,0])
        #returns[:, i] = np.divide(data[fxnames[i]][:,:1]-data[fxnames[i]][:,0], data[fxnames[i]][:,0])
        
    #spreads 
    spreads = np.square(returns)
    
    #binary up downs
    ups = (returns>0).astype(np.int_)
    downs = (returns<0).astype(np.int_)
    
    #what time during a day 
    timeframe = np.zeros((data[fxnames[i]].shape[0]-1, 1))
    timeframe[:, 0] = hours[fxnames[i]][:-1]/6.0
         
    final = np.concatenate([returns, spreads, timeframe], axis=1) #ignore binary ups or downs for now
    return final, data, times

def generate_sampleindex(data, target=0, th=2, barrier=20, paststeps=300, futuresteps=1, factor=12):
    majors = ["USDJPY", "USDCHF", "USDCAD", "NZDUSD", "GBPUSD", "EURUSD", "AUDUSD"]
    print "Predicting the movement of", majors[0]
    
    upindex = [i for i in range(data.shape[0]) if data[i, target] > th and i > paststeps*factor+futuresteps]  
    downindex = [i for i in range(data.shape[0]) if data[i, target] < -1*th and i > paststeps*factor+futuresteps] 
    
    mask = np.ones(data.shape[0]).astype(np.bool_)
    for i in upindex+downindex:
        for j in range(max(i-barrier, 0), min(i+barrier, data.shape[0])):
            mask[j] = False
    
    negativeindex = np.arange(data.shape[0])[mask]
    np.random.shuffle(negativeindex)
    negativeindex = [i for i in negativeindex if i > paststeps*factor+futuresteps]
    negativeindex = negativeindex[:(len(upindex)+len(downindex))/2]
    negativeindex.sort()
    
    return upindex, downindex, negativeindex

def aggregate_samples(X, factor=12):
    assert X.shape[0]%factor == 0
    ret = np.zeros((X.shape[0]/factor, X.shape[1]))
    for i in range(ret.shape[0]):
        ret[i] = np.mean(X[i*factor:(i+1)*factor, :], axis=0)
    return ret

def generate_samples(data, indexlist, target=0, paststeps=300, futuresteps=1, factor1=12, factor2=36, suffix=""):
    label = ["up", "down", "negative"]
    count2 =0
    Xs =[]
    X2s = []
    X3s = []
    Ys = []
    YRs = []
    
    for indexes in indexlist:
        X = np.zeros((len(indexes), paststeps, data.shape[1]))
        X2 = np.zeros((len(indexes), paststeps, data.shape[1]))
        X3 = np.zeros((len(indexes), paststeps, data.shape[1]))
        Y = np.zeros((len(indexes), 3))
        YR = np.zeros(len(indexes))
        Y[:, count2] = 1
        count2 += 1
        
        count = 0
        for i in indexes:
            X[count] = data[i-paststeps-futuresteps+1:i-futuresteps+1]
            X2[count] = aggregate_samples(data[i-paststeps*factor1-futuresteps+1:i-futuresteps+1], factor1)
            X3[count] = aggregate_samples(data[i-paststeps*factor2-futuresteps+1:i-futuresteps+1], factor2)
            YR[count] = data[i][target]
            count += 1
        Xs.append(X)
        X2s.append(X2)
        X3s.append(X3)
        Ys.append(Y)
        YRs.append(YR)
    
    print "X shape:", X.shape
    
    np.save("Y"+suffix, np.concatenate(Ys, axis=0))
    np.save("YR"+suffix, np.concatenate(YRs, axis=0))
    np.save("X"+suffix, np.concatenate(Xs, axis=0))
    np.save("X2"+suffix, np.concatenate(X2s, axis=0))
    np.save("X3"+suffix, np.concatenate(X3s, axis=0))
        
def main():
    #try:
    majors = ["USDJPY", "USDCHF", "USDCAD", "NZDUSD", "GBPUSD", "EURUSD", "AUDUSD"]
    prefix = sys.argv[1]#"../data/forex_majors_candle/"
    suffix = sys.argv[2]#"-2017_06_01-2017_06_26.csv"
    testnames = [prefix+m+suffix for m in majors]

    #Generate data points
    # 12 intervals = 60 mins/5 mins
    _5min_final, _5mindata, times = process_data(testnames, majors, 12)

    # Normalize
    _5min_scaled = preprocessing.scale(_5min_final)
    
    # Sanity check
    #print np.mean(_5min_scaled, axis=0)[:10]
    #print np.std(_5min_scaled, axis=0)
    #print len([i for i in np.std(_5min_scaled, axis=0) if i==0])

    # Generate index
    # 1/36 resolution because we are looking for 3 hours (36 intervals) pers step.
    up, down, neg = generate_sampleindex(_5min_scaled, th=2, factor=36)

    # Generate samples
    generate_samples(_5min_final, [up, down, neg], suffix=suffix[:-4])
    #except:
    #    print "python generate_data.py ../data/forex_majors_candle/5m -2017_06_01-2017_06_26.csv"

if __name__ == "__main__":
    main()
