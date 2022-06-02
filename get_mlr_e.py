#!/usr/bin/env python

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from sklearn.preprocessing import MaxAbsScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import cm


df = pd.read_csv("DA_Th_Ur_Sq_MLR_all.csv")
print(df.head())
# print(df.head())
columns = list(df.columns)

# Fix targets
DeltaG_3 = df.DeltaG_3.to_numpy(dtype=float)
log_TOF = df.logTOF.to_numpy(dtype=float)


# Without scaling, just in case
# d1 = np.squeeze((df.NCBG_LUMO.to_numpy().reshape(-1,1)))
# d2 = np.squeeze((df.BB_wL.to_numpy().reshape(-1,1)))
# d3 = np.squeeze((df.BB_iNH.to_numpy().reshape(-1,1)))
# d4 = np.squeeze((df.CBG_ENF.to_numpy().reshape(-1,1)))
# d5 = np.squeeze((df.CBG_Nk.to_numpy().reshape(-1,1)))
# d6 = np.squeeze((df.NCBG_Hardness.to_numpy().reshape(-1,1)))
# d7 = np.squeeze((df.CBG_LUMO.to_numpy().reshape(-1,1)))
# d8 = np.squeeze((df.BB_NBO_N.to_numpy().reshape(-1,1)))
# d9 = np.squeeze((df.NCBG_PEX.to_numpy().reshape(-1,1)))

# Selected descriptor matrix
d1 = np.squeeze(MaxAbsScaler().fit_transform(df.NCBG_LUMO.to_numpy().reshape(-1, 1)))
d2 = np.squeeze(MaxAbsScaler().fit_transform(df.BB_wL.to_numpy().reshape(-1, 1)))
d3 = np.squeeze(MaxAbsScaler().fit_transform(df.BB_iNH.to_numpy().reshape(-1, 1)))
d4 = np.squeeze(MaxAbsScaler().fit_transform(df.CBG_ENF.to_numpy().reshape(-1, 1)))
d5 = np.squeeze(MaxAbsScaler().fit_transform(df.CBG_Nk.to_numpy().reshape(-1, 1)))
d6 = np.squeeze(
    MaxAbsScaler().fit_transform(df.NCBG_Hardness.to_numpy().reshape(-1, 1))
)
d7 = np.squeeze(MaxAbsScaler().fit_transform(df.CBG_LUMO.to_numpy().reshape(-1, 1)))
d8 = np.squeeze(MaxAbsScaler().fit_transform(df.BB_NBO_N.to_numpy().reshape(-1, 1)))
d9 = np.squeeze(MaxAbsScaler().fit_transform(df.NCBG_PEX.to_numpy().reshape(-1, 1)))


# The regression features used in the manuscript
d_manuscript = np.c_[d1, d2, d3, d4, d5]
h_manuscript = [4, 23, 27, 20, 19]

# An alternative 5-parameter model 1
d_alternative_1 = np.c_[d2, d4, d6, d7, d8]
h_alternative_1 = [23, 20, 7, 14, 28]

# An alternative 5-parameter model 2
d_alternative_2 = np.c_[d9, d2, d3, d4, d5]
h_alternative_2 = [12, 23, 27, 20, 19]

# A model using three additional features
d_increased = np.c_[d1, d2, d3, d4, d5, d6, d7, d8]
h_increased = [4, 23, 27, 20, 19, 7, 14, 28]

# Models using reduced features
d_reduced_1 = np.c_[d1, d2, d3, d4]
h_reduced_1 = [4, 23, 27, 20]
d_reduced_2 = np.c_[d2, d3, d4, d5]
h_reduced_2 = [23, 27, 20, 19]


# Model using about half of all features (12)
d_almostall = MaxAbsScaler().fit_transform(df.to_numpy()[:, 3:27:2])
h_almostall = range(0, 27)[3:27:2]

# Model using all features (26)
d_all = MaxAbsScaler().fit_transform(df.to_numpy()[:, 3:])
h_all = range(3, 29)

df_scaled = pd.DataFrame(
    data=np.c_[df.Label.to_numpy(dtype=object), DeltaG_3, log_TOF, d_all],
    index=np.arange(101),
    columns=columns,
)
df_scaled.to_csv("DA_Th_Ur_Sq_MLR_all_scaled.csv", index=False)

model_parameters = [
    d_manuscript,
    d_alternative_1,
    d_alternative_2,
    d_increased,
    d_reduced_1,
    d_reduced_2,
    d_almostall,
    d_all,
]
model_column_headers = [
    h_manuscript,
    h_alternative_1,
    h_alternative_2,
    h_increased,
    h_reduced_1,
    h_reduced_2,
    h_almostall,
    h_all,
]
model_names = [
    "Manuscript model",
    "Alternative 5-parameter model 1",
    "Alternative 5-parameter model 2 (with PEX instead of LUMO)",
    "8-parameter model",
    "4-parameter model 1",
    "4-parameter model 2",
    "12-parameter model",
    "26-parameter model",
]


# A helper function to check the regression quality
def check_reg(d, func, popt, e):
    e_test = func(
        d,
        *popt,
    )

    slope, intercept, r_value, p_value, std_err = stats.linregress(e_test, e)
    factor = (d.shape[0] - 1) / (d.shape[0] - d.shape[1])
    print(
        "\n============================================\nMAE with combined function {0} with r2 of {1} and adjusted r2 of {2}\n============================================".format(
            np.round(np.abs(e - e_test).mean(), 2),
            np.round(r_value**2, 2),
            np.round(1 - (1 - r_value**2) * factor, 2),
        )
    )


# A general helper for linear models
def check_predictions(e_test, e, verb=0):
    slope, intercept, r_value, p_value, std_err = stats.linregress(e_test, e)
    r2 = r_value**2
    factor = (d.shape[0] - 1) / (d.shape[0] - d.shape[1])
    mae = abs(e_test - e).mean()
    if verb > 0:
        print(
            "\n============================================\nMAE with combined function {0} with r2 of {1} and adjusted r2 of {2}\n============================================".format(
                np.round(mae, 2),
                np.round(r2, 2),
                np.round(1 - (1 - r_value**2) * factor, 2),
            )
        )
    return mae, r2


# 4-parameter MLR model
def mvr4(d, a0, a1, a2, a3, offset):
    coeffs = np.array([a0, a1, a2, a3])
    t = np.multiply(d[:], coeffs[:]).sum(axis=1)
    return t + offset


# 5-parameter MLR model
def mvr5(d, a0, a1, a2, a3, a4, offset):
    coeffs = np.array([a0, a1, a2, a3, a4])
    t = np.multiply(d[:], coeffs[:]).sum(axis=1)
    # Sanity check without vectorizing nicely
    # return np.array( a0 * d[:,0] + a1 * d[:,1] + a2 * d[:,2] + a3 * d[:,3] + a4 * d[:,4] + offset )
    return t + offset


# 8-parameter MLR model
def mvr8(d, a0, a1, a2, a3, a4, a5, a6, a7, offset):
    coeffs = np.array([a0, a1, a2, a3, a4, a5, a6, a7])
    t = np.multiply(d[:], coeffs[:]).sum(axis=1)
    return t + offset


# 12-parameter MLR model
def mvr12(
    d,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    offset,
):
    coeffs = np.array([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11])
    t = np.multiply(d[:], coeffs[:]).sum(axis=1)
    return t + offset


# 26-parameter MLR model
def mvr26(
    d,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15,
    a16,
    a17,
    a18,
    a19,
    a20,
    a21,
    a22,
    a23,
    a24,
    a25,
    offset,
):
    coeffs = np.array(
        [
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            a10,
            a11,
            a12,
            a13,
            a14,
            a15,
            a16,
            a17,
            a18,
            a19,
            a20,
            a21,
            a22,
            a23,
            a24,
            a25,
        ]
    )
    t = np.multiply(d[:], coeffs[:]).sum(axis=1)
    return t + offset


for d, h, name in zip(model_parameters, model_column_headers, model_names):
    e = DeltaG_3
    print(
        f"\n============================================\nAttempting to fit model {name} to DeltaG_3. Replace line below this message for TOF fitting.\n============================================"
    )
    # print(f"Data snippet:\n {d[0:5,:]}\n")
    if name == "Manuscript model":
        popt0 = [
            4.17822436,
            -3.33662529,
            -1.77303116,
            10.45749966,
            1.8136876,
            -3.12777498,
        ]
        popt0, pcov0 = sp.optimize.curve_fit(mvr5, d, e, method="dogbox", p0=popt0)
        check_reg(d, mvr5, popt0, e)
    elif d.shape[1] == 5:
        try:
            popt0, pcov0 = sp.optimize.curve_fit(mvr5, d, e, method="trf")
        except:
            popt0, pcov0 = sp.optimize.curve_fit(mvr5, d, e, method="dogbox")
        check_reg(d, mvr5, popt0, e)
    elif d.shape[1] == 4:
        try:
            popt0, pcov0 = sp.optimize.curve_fit(mvr4, d, e, method="trf")
        except:
            popt0, pcov0 = sp.optimize.curve_fit(mvr4, d, e, method="dogbox")
        check_reg(d, mvr4, popt0, e)
    elif d.shape[1] == 8:
        try:
            popt0, pcov0 = sp.optimize.curve_fit(mvr8, d, e, method="trf")
        except:
            popt0, pcov0 = sp.optimize.curve_fit(mvr8, d, e, method="dogbox")
        check_reg(d, mvr8, popt0, e)
    elif d.shape[1] == 12:
        try:
            popt0, pcov0 = sp.optimize.curve_fit(mvr12, d, e, method="trf")
        except:
            popt0, pcov0 = sp.optimize.curve_fit(mvr12, d, e, method="dogbox")
        check_reg(d, mvr12, popt0, e)
    elif d.shape[1] == 26:
        try:
            popt0, pcov0 = sp.optimize.curve_fit(mvr26, d, e, method="trf")
        except:
            popt0, pcov0 = sp.optimize.curve_fit(mvr26, d, e, method="dogbox")
        check_reg(d, mvr26, popt0, e)

    for i, feature_name in enumerate(h):
        print(f"Parameter {columns[feature_name]} with coefficient {round(popt0[i],4)}")
    print(f"and offset {popt0[-1]}")
