Preprocessing:
1)highpass 0.01 Butterworths order 7 IIR
2)lowpass 0.1 Butterworths order 1 IIR

Scaled correlation analysis
- Scale Correlation +-20s, scale = 20s
- Exported:
*) Value at Max Absolute SCA value and time offset (interpolated smooth SCA - Akima interpolation)
	- The SCAMaxAbs values are validated. For validation two thresholds are computed:
			ThLo = Q2 - k*IRQ
			ThHi = Q2 + k*IRQ
		Only the values below ThLo or above ThHi are considered valid.
	- there will be several values for K.
*) Pearson R computed with SCA: SCA scale = trial length 
*) SCA at lag 0

Filename are encoded with relevant fields separated by '_' and '-' as follows:
PIPELINE_FEATURE_VALIDATION-ANIMALID-CONDITION-TRIALID-VALUEOFFESTDESCRIPTION.EXTENSION
Note: VALIDATION ins missing for features that do not have one