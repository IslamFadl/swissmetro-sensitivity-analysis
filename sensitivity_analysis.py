# -*- coding: utf-8 -*-
"""
Dies ist eine temporäre Skriptdatei.
"""

"""

Logit model
===========

Estimation of a logit model with several algorithms.

:author: Michel Bierlaire, EPFL
:date: Tue Nov  7 17:00:09 2023

"""
#pip install biogeme   # Install biogeme package via Console

import itertools
import pandas as pd

from biogeme.parameters import Parameters
from biogeme.tools import format_timedelta
import biogeme.biogeme as bio
from biogeme import models
import biogeme.exceptions as excep
from biogeme.expressions import Beta

import biogeme.database as db
from biogeme.expressions import Variable

df_swissmetro = pd.read_csv('./swissmetro.dat', sep='\t')

database = db.Database('swissmetro', df_swissmetro)

GROUP = Variable('GROUP')
SURVEY = Variable('SURVEY')
SP = Variable('SP')
ID = Variable('ID')
PURPOSE = Variable('PURPOSE')
FIRST = Variable('FIRST')
TICKET = Variable('TICKET')
WHO = Variable('WHO')
LUGGAGE = Variable('LUGGAGE')
AGE = Variable('AGE')
MALE = Variable('MALE')
INCOME = Variable('INCOME')
GA = Variable('GA')
ORIGIN = Variable('ORIGIN')
DEST = Variable('DEST')
TRAIN_AV = Variable('TRAIN_AV')
CAR_AV = Variable('CAR_AV')
SM_AV = Variable('SM_AV')
TRAIN_TT = Variable('TRAIN_TT')
TRAIN_CO = Variable('TRAIN_CO')
TRAIN_HE = Variable('TRAIN_HE')
SM_TT = Variable('SM_TT')
SM_CO = Variable('SM_CO')
SM_HE = Variable('SM_HE')
SM_SEATS = Variable('SM_SEATS')
CAR_TT = Variable('CAR_TT')
CAR_CO = Variable('CAR_CO')
CHOICE = Variable('CHOICE')

remove = (((database.data.PURPOSE != 1) &
         (database.data.PURPOSE != 3)) |
         (database.data.CHOICE == 0))
database.data.drop(database.data[remove].index,inplace=True)

exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)

SM_COST = database.DefineVariable('SM_COST', SM_CO * (GA == 0))
TRAIN_COST = database.DefineVariable('TRAIN_COST', TRAIN_CO * (GA == 0))
CAR_AV_SP = database.DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0))
TRAIN_AV_SP = database.DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))
TRAIN_TT_SCALED = database.DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100)
TRAIN_COST_SCALED = database.DefineVariable('TRAIN_COST_SCALED', TRAIN_COST / 100)
SM_TT_SCALED = database.DefineVariable('SM_TT_SCALED', SM_TT / 100)
SM_COST_SCALED = database.DefineVariable('SM_COST_SCALED', SM_COST / 100)
CAR_TT_SCALED = database.DefineVariable('CAR_TT_SCALED', CAR_TT / 100)
CAR_CO_SCALED = database.DefineVariable('CAR_CO_SCALED', CAR_CO / 100)



# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Definition of the utility functions.
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b01logit'
#the_biogeme.calculate_null_loglikelihood(av)

results = the_biogeme.estimate()
print(results.short_summary())
pandas_results = results.getEstimatedParameters()
print(pandas_results)
