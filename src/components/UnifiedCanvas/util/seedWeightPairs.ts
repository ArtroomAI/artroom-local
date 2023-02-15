/* eslint-disable no-mixed-spaces-and-tabs */
import { SeedWeights, SeedWeightPair } from '../painter';

export const stringToSeedWeights = (string: string): SeedWeights | boolean => {
	const stringPairs = string.split(',');
	const arrPairs = stringPairs.map(p => p.split(':'));
	const pairs = arrPairs.map((p: Array<string>): SeedWeightPair => {
		return { seed: Number(p[0]), weight: Number(p[1]) };
	});

	if (!validateSeedWeights(pairs)) {
		return false;
	}

	return pairs;
};

export const validateSeedWeights = (
	seedWeights: SeedWeights | string,
): boolean => {
	return typeof seedWeights === 'string'
		? Boolean(stringToSeedWeights(seedWeights))
		: Boolean(
				seedWeights.length &&
					!seedWeights.some((pair: SeedWeightPair) => {
						const { seed, weight } = pair;
						const isSeedValid = !isNaN(
							parseInt(seed.toString(), 10),
						);
						const isWeightValid =
							!isNaN(parseInt(weight.toString(), 10)) &&
							weight >= 0 &&
							weight <= 1;
						return !(isSeedValid && isWeightValid);
					}),
		  );
};

export const stringToSeedWeightsArray = (
	string: string,
): Array<Array<number>> => {
	const stringPairs = string.split(',');
	const arrPairs = stringPairs.map(p => p.split(':'));
	return arrPairs.map(
		(p: Array<string>): Array<number> => [parseInt(p[0]), parseFloat(p[1])],
	);
};
