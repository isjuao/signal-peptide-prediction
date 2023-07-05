import re
import math

# extract patterns - without counts
"""
patterns = ['SSS', 'T']
text = 'IIIIISSSIIII'
for pattern in patterns:
    if re.search(pattern, text):
        print('ist drin!')
    else:
        print('nicht drin')
"""

# extract patterns - counts
def parse(fastafile, outfile):
    motif = "SSS"
    header = None
    with open(fastafile, 'r') as inputfile, open(outfile, 'w') as outputfile:
            linecounter = 0
            for line in inputfile:
                if line.startswith('>'):
                    if header is not None:
                        outputfile.write(header + '\t' + str(count) + '\t' + motif + '\n')
                    header = line
                    count = 0
                    linecounter = 0
                elif linecounter == 2:
                    matches = re.findall(motif, line)
                    count += len(matches)
                linecounter += 1
            if header is not None:
                outputfile.write(header + '\t' + str(count) + '\t' + motif + '\n')


''' parse("train_set.fasta", "motifs.xls") '''

# analysis of file (percentages etc)
def analyzePerResidue(fastafile, isTrainSet):
    setName = ""
    if isTrainSet == True:
        setName = "TrainSet"
    else:
        setName = "BenchmarkSet"
    inputfile = open(fastafile, 'r')
    linecounter = 0
    noOfResidues = 0
    specialResidues = [0]*6
    globalLinecounter = 0
    splitStatistic = [[0 for j in range(6)] for i in range(5)]
    splitOfLine = 0
    domainStatistic = [[0 for j in range(6)] for i in range(4)]
    domainOfLine = 0
    typeStatistic = [[0 for j in range(6)] for i in range(4)]
    typeOfLine = 0
    lengthOfSPArray = [[0 for j in range(1)] for i in range(3)]
    for line in inputfile:
        if line.startswith('>'):
            linecounter = 0
            # find split
            splitOfLine = int(line.strip()[-1:])
            # find domain of life
            domainName = line[8:]
            if domainName.startswith("EUKARYA"):
                domainOfLine = 0
            elif domainName.startswith("ARCHAEA"):
                domainOfLine = 1
            elif domainName.startswith("POSITIVE"):
                domainOfLine = 2
            elif domainName.startswith("NEGATIVE"):
                domainOfLine = 3
            # find type of protein
            if "SP" in line and line.find("SP") > 7 and line.find("NO_SP") == -1:
                typeOfLine = 0
            elif "NO_SP" in line:
                typeOfLine = 1
            elif "TAT" in line and line.find("TAT") > 7:
                typeOfLine = 2
            elif "LIPO" in line and line.find("LIPO") > 7:
                typeOfLine = 3
        elif linecounter == 2:
            '''if globalLinecounter == 42110:
                for char in line:
                    print("'" + char + "'")'''
            lengthOfSP_S = 0
            lengthOfSP_T = 0
            lengthOfSP_L = 0
            for char in line:
                if char is "S":
                    specialResidues[0] += 1
                    splitStatistic[splitOfLine][0] += 1
                    domainStatistic[domainOfLine][0] += 1
                    typeStatistic[typeOfLine][0] += 1
                    lengthOfSP_S += 1
                elif char is "T":
                    specialResidues[1] += 1
                    splitStatistic[splitOfLine][1] += 1
                    domainStatistic[domainOfLine][1] += 1
                    typeStatistic[typeOfLine][1] += 1
                    lengthOfSP_T += 1
                elif char is "L":
                    specialResidues[2] += 1
                    splitStatistic[splitOfLine][2] += 1
                    domainStatistic[domainOfLine][2] += 1
                    typeStatistic[typeOfLine][2] += 1
                    lengthOfSP_L += 1
                elif char is "I":
                    specialResidues[3] += 1
                    splitStatistic[splitOfLine][3] += 1
                    domainStatistic[domainOfLine][3] += 1
                    typeStatistic[typeOfLine][3] += 1
                elif char is "M":
                    specialResidues[4] += 1
                    splitStatistic[splitOfLine][4] += 1
                    domainStatistic[domainOfLine][4] += 1
                    typeStatistic[typeOfLine][4] += 1
                elif char is "O":
                    specialResidues[5] += 1
                    splitStatistic[splitOfLine][5] += 1
                    domainStatistic[domainOfLine][5] += 1
                    typeStatistic[typeOfLine][5] += 1
                '''else:
                    print(char)'''
            if(lengthOfSP_L != 0):
                lengthOfSPArray[2].append(lengthOfSP_L)
            if(lengthOfSP_S != 0):
                lengthOfSPArray[0].append(lengthOfSP_S)
            if(lengthOfSP_T != 0):
                lengthOfSPArray[1].append(lengthOfSP_T)
            noOfResidues += len(line) - 1
        linecounter += 1
        globalLinecounter += 1
    # fail: print("Total number of residues (with backslash/n): " + str(noOfResidues))'
    nameArray = ["S", "T", "L", "I", "M", "O"]
    checkSum = 0
    for i in range(0, 6):
        checkSum += specialResidues[i]
    print("[   " + setName + " global:   ]\n")
    for i in range(0, 6):
        print(nameArray[i] + ": " + str(specialResidues[i]) + " = "
              + str(math.ceil(specialResidues[i]/checkSum*1000)/1000))
    print("Total number of residues: " + str(checkSum) + '\n\n')

    # splits
    totalInSplit = [0]*5
    for i in range(0, 5):
       for j in range(0, 6):
            totalInSplit[i] += splitStatistic[i][j]
    print("[   " + setName + " splits:   ]")
    for i in range(0, 5):
        splitChecksumRatio = math.ceil(totalInSplit[i]/checkSum * 1000) / 1000
        print("\n[  split " + str(i) + " (" + str(splitChecksumRatio) + ")  ]")
        # control: checkSumPerSplit = 0
        allPercentages = 0 # should be 1.0
        for j in range(0, 6):
            # control: checkSumPerSplit += splitStatistic[i][j]
            percentage = math.ceil(splitStatistic[i][j]/totalInSplit[i] * 1000) / 1000
            print(nameArray[j] + ": " + str(splitStatistic[i][j]) + " = "
                  + str(percentage))
            allPercentages += percentage
        # control: print(str(totalInSplit[i]) + " == " + str(checkSumPerSplit))
        print("total in split = " + str(totalInSplit[i]))
        print("control: total percentages = " + str(math.ceil(allPercentages * 1000) / 1000))

    # domains
    totalInDomain = [0]*4
    domainNameArray = ["EUKARYA", "ARCHAEA", "POSITIVE", "NEGATIVE"]
    for i in range(0, 4):
        for j in range(0, 6):
            totalInDomain[i] += domainStatistic[i][j]
    print("\n\n" + "[   " + setName + " domains:   ]")
    for i in range(0, 4):
        domainChecksumRatio = math.ceil(totalInDomain[i]/checkSum * 1000) / 1000
        print("\n[  domain " + domainNameArray[i] + " (" + str(domainChecksumRatio) + ")  ]")
        allPercentages = 0
        for j in range(0, 6):
            percentage = math.ceil(domainStatistic[i][j] / totalInDomain[i] * 1000) / 1000
            print(nameArray[j] + ": " + str(domainStatistic[i][j]) + " = " + str(percentage))
            allPercentages += percentage
        print("total in domain = " + str(totalInDomain[i]))
        print("control: total percentages = " + str(math.ceil(allPercentages * 1000) / 1000))

    # types
    totalInType = [0] * 4
    typeNameArray = ["SP", "NO_SP", "TAT", "LIPO"]
    storingArrayAllSP = [0] * 6
    for i in range(0, 4):
        for j in range(0, 6):
            totalInType[i] += typeStatistic[i][j]
    print("\n\n" + "[   " + setName + " types:   ]")
    for i in range(0, 4):
        typeCheckSumRatio = math.ceil(totalInType[i] / checkSum * 1000) / 1000
        print("\n[  type " + typeNameArray[i] + " (" + str(typeCheckSumRatio) + ")  ]")
        allPercentages = 0
        for j in range(0, 6):
            percentage = math.ceil(typeStatistic[i][j] / totalInType[i] * 1000) / 1000
            if i == 0 or i == 2 or i == 3:
                storingArrayAllSP[j] += typeStatistic[i][j]
            print(nameArray[j] + ": " + str(typeStatistic[i][j]) + " = " + str(percentage))
            allPercentages += percentage
        print("total in type = " + str(totalInType[i]))
        print("control: total percentages = " + str(math.ceil(allPercentages * 1000) / 1000))

    # proteins with SPs
    print("\n[   all SP types (SP, TAT, LIPO):   ]")
    totalWithSP = totalInType[0] + totalInType[2] + totalInType[3]
    allPercentages = 0
    print(str(totalWithSP) + " = " + str(math.ceil(totalWithSP/checkSum*1000)/1000) + " of all proteins have SPs\n"
                                                           "statistics for the group of proteins that carry a SP:")
    for i in range(0, 6):
        percentage = math.ceil(storingArrayAllSP[i] / totalWithSP * 1000) / 1000
        print(nameArray[i] + ": " + str(storingArrayAllSP[i]) + " = " + str(percentage))
        allPercentages += percentage
    print("control: total percentages = " + str(math.ceil(allPercentages * 1000) / 1000))
    typeOfSPNameArray = ["Sec/SPI(SP)", "Tat/SPI(TAT)", "Sec/SPII(LIPO)"]
    print("\n[ average SP length ]")
    for i in range(0, 3):
        averageLength = 0
        for j in range(0, len(lengthOfSPArray[i])):
            averageLength += lengthOfSPArray[i][j]
        averageLength = averageLength/len(lengthOfSPArray[i])
        print(str(typeOfSPNameArray[i]) + ": average length " + str(int(round(averageLength))))


'''analyzePerResidue("train_set.fasta", True)
print("\n\n---------------------\n")
analyzePerResidue("benchmark_set.fasta", False)'''

def analyzePerDomain(fastafile, isTrainSet):
    setName = ""
    if isTrainSet == True:
        setName = "TrainSet"
    else:
        setName = "BenchmarkSet"
    inputfile = open(fastafile, 'r')
    linecounter = 0
    domainName = ""
    domainOfLine = 0
    domainNameArray = ["EUKARYA", "ARCHAEA", "POSITIVE", "NEGATIVE"]
    typeOfLine = 0
    typeNameArray = ["SP", "NO_SP", "TAT", "LIPO"]
    domainTypeStatistic = [[0 for j in range(4)] for i in range(4)]     # row = domain, column = type
    for line in inputfile:
        if line.startswith('>'):
            linecounter += 1
            # find domain
            domainName = line[8:]
            if domainName.startswith("EUKARYA"):
                domainOfLine = 0
            elif domainName.startswith("ARCHAEA"):
                domainOfLine = 1
            elif domainName.startswith("POSITIVE"):
                domainOfLine = 2
            elif domainName.startswith("NEGATIVE"):
                domainOfLine = 3
            # find type
            if "SP" in line and line.find("SP") > 7 and line.find("NO_SP") == -1:
                typeOfLine = 0
            elif "NO_SP" in line:
                typeOfLine = 1
            elif "TAT" in line and line.find("TAT") > 7:
                typeOfLine = 2
            elif "LIPO" in line and line.find("LIPO") > 7:
                typeOfLine = 3
            domainTypeStatistic[domainOfLine][typeOfLine] += 1
    print("[   " + setName + " types   ]")
    for j in range(0, 4):
        sumOfType = 0
        for i in range(0, 4):
            sumOfType += domainTypeStatistic[i][j]
        print(typeNameArray[j] + ": " + str(sumOfType) + "  (= " +
              str(math.ceil(sumOfType/linecounter * 1000)/1000) + ")")
    totalInDomains = [0]*4
    print("\n[   " + setName + " domains   ]")
    for i in range(0, 4):
        sumOfDomain = 0
        for j in range(0, 4):
            sumOfDomain += domainTypeStatistic[i][j]
        totalInDomains[i] = sumOfDomain
        print(domainNameArray[i] + ": " + str(sumOfDomain) + "  (= " +
              str(math.ceil(sumOfDomain/linecounter * 1000)/1000) + ")")
    for i in range(0, 4):
        print("\n[   Domain " + domainNameArray[i] + "   ]")
        for j in range(0, 4):
            print(typeNameArray[j] + ": " + str(domainTypeStatistic[i][j]) + "  (= " +
              str(math.ceil(domainTypeStatistic[i][j]/totalInDomains[i] * 1000)/1000) + ")")


'''print("\n//\n")
analyzePerDomain("train_set.fasta", True)
print("\n\n---------------------\n")
analyzePerDomain("benchmark_set.fasta", False)'''

def parseLengthsIntoFile(fastafile, isTrainSet, outfile):
    setName = ""
    if isTrainSet == True:
        setName = "TrainSet"
    else:
        setName = "BenchmarkSet"
    with open(fastafile, 'r') as inputfile, open(outfile, 'w') as outputfile:
        lengthOfSPArray = [[0 for j in range(0)] for i in range(3)]
        nameArray = ["SP", "TAT", "LIPO"]
        linecounter = 0
        for line in inputfile:
            if line.startswith('>'):
                linecounter = 0
            elif linecounter == 2:
                lengthOfSP_S = 0
                lengthOfSP_T = 0
                lengthOfSP_L = 0
                for char in line:
                    if char is "S":
                        lengthOfSP_S += 1
                    elif char is "T":
                        lengthOfSP_T += 1
                    elif char is "L":
                        lengthOfSP_L += 1
                if (lengthOfSP_L != 0):
                    lengthOfSPArray[2].append(lengthOfSP_L)
                elif (lengthOfSP_S != 0):
                    lengthOfSPArray[0].append(lengthOfSP_S)
                elif (lengthOfSP_T != 0):
                    lengthOfSPArray[1].append(lengthOfSP_T)
            linecounter += 1
        for i in range(0, 3):
            for j in range(0, len(lengthOfSPArray[i])):
                outputfile.write(nameArray[i] + '\t' + str(lengthOfSPArray[i][j]) + '\n')


'''parseLengthsIntoFile("train_set.fasta", True, "sptypes.xls")'''
parseLengthsIntoFile("benchmark_set.fasta", False, "sptypes.xls")
