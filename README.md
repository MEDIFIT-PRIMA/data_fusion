# Data Fusion Models

## Available Fusion Models Description
- c_f1_ad.fskx: Model for the detection of adulteration in cheese samples (FT-NIR x FT-MIR sensor fusion).
- c_f1_geo.fskx: Nodel for the geographical origin detection in cheese samples (FT-NIR x FT-MIR sensor fusion).
- c_f2_ad.fskx: Model for the detection of adulteration in cheese samples (FT-NIR x FT-MIR x CRM sensor fusion).
- c_f2_geo.fskx: Model for the geographical origin detection in cheese samples (FT-NIR x FT-MIR x CRM sensor fusion).
- h_f1_ad.fskx: Model for the detection of adulteration in honey samples (FT-NIR x FT-MIR sensor fusion).
- h_f1_bot.fskx: Model for the botanical origin detection in cheese samples (FT-NIR x FT-MIR sensor fusion).
- h_f1_geo.fskx: Model for the geographical origin detection in cheese samples (FT-NIR x FT-MIR sensor fusion).
- h_f2_ad.fskx: Model for the detection of adulteration in honey samples (FT-NIR x FT-MIR x CRM sensor fusion).
- h_f2_bot.fskx: Model for the botanical origin detection in cheese samples (FT-NIR x FT-MIR x CRM sensor fusion).
- h_f2_geo.fskx: Model for the geographical origin detection in cheese samples (FT-NIR x FT-MIR x CRM sensor fusion).

## Notation for the Fusion Models' names
- h: honey 
- c: cheese
- f1: Fusion 1 between FTNIR and FTMIR sensors
- f2: Fusion 2 between FTNIR, FTMIR and CRM sensors
- ad: authentication model
- geo: geographical origin model
- bot: botanical origin model

## Required Input Parameters:
- f1 models: For models that belong to F1 fusion scenario (FT-NIR x FT-MIR sensor fusion) two discrete excel files with the spectral responses of the two sensors (one sample per row) are required. The excel file names must contain the sensor's name ("ftnir" or "ftmir", for FT-NIR and FT-MIR respectively).
- f2 models: For models that belong to F2 fusion scenario (FT-NIR x FT-MIR x CRM sensor fusion) the required inputs include the same as in f1 models with the addition of an extra excel file for the CRM data. The excel file name for the CRM data must contain the sensor's name ("raman").
- Example data for every fusion scenario are included in the ExampleData file

## Expected Output Description:
- 'ad': Adulterated Sample
- 'es': Spanish Sample
- 'gr': Greek Sample
- 'mt': Maltese Sample
- 'tn': Tunisia Sample
- 'tr': Turkish Sample
- 'th': Thyme Sample (Honey Specific)
- 'pi': Pine Sample (Honey Specific)
- 'pf': Polyfloral Sample (Honey Specific)
