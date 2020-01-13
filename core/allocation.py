def allocationFun(x):
    return MVF(h, omega, betas, failures)

def MVF(h, omega, betas, failures):
    # can clean this up to use less loops, probably
    covData = np.array(self.covariateData)
    for i in range(x):
        covData.append(x[i]) 

    prodlist = []
    for i in range(failures + 1):     # CHANGED THIS FROM self.n + 1 !!!
        sum1 = 1
        sum2 = 1
        TempTerm1 = 1
        for j in range(self.numCovariates):
                TempTerm1 = TempTerm1 * np.exp(self.covariateData[j][i] * betas[j])
        sum1 = 1-((1 - h[i]) ** (TempTerm1))
        for k in range(i):
            TempTerm2 = 1
            for j in range(self.numCovariates):
                    TempTerm2 = TempTerm2 * np.exp(self.covariateData[j][k] * betas[j])
            sum2 = sum2 * ((1 - h[i])**(TempTerm2))
        prodlist.append(sum1 * sum2)
    return omega * sum(prodlist)