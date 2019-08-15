# from tool.py
raw_imported_data = []  # raw imported data from csv, list of strings

# ---- from covariate.py ----
failure_times = []
kVec_cumulative = []
mvf_list = []
llf_val = 0.0
aic_val = 0.0
bic_val = 0.0
sse_val = 0.0
cov_data = []
kVec = []
n = 0
total_failures = 0
num_covariates = 0
# model fitting
b = 0
betas = []
geometric_hazard = []
nb2_hazard = []
dw2_hazard = []
nb_hazard = []
dw_hazard = []
h = []

# flags
has_data = False        # flag stores if data has been loaded
estimation_ran = False  # has estimation been run yet

# unused/removed stuff
if __name__ == "__main__":
    pass