import { ImageLayer } from '../components/UnifiedCanvas/atoms/canvasTypes';

export const LAYER_REG_EXP = /Layer\s\((\d)\)/;
export const MERGED_LAYER_REG_EXP = /Merged layer\s\((\d)\)/;

export const getNameForLayer = (
	query?: RegExp,
	layers?: ImageLayer[],
	coreName = 'Layer',
) => {
	if (layers && query) {
		let maxIndex = 1;
		const filteredNames = layers
			.filter(elem => query.test(elem.name))
			.map(elem => elem.name);

		filteredNames.forEach(elem => {
			const data = query.exec(elem);
			const parsedIndex = parseInt(data?.[1] || '0');
			if (data && maxIndex < parsedIndex) {
				maxIndex = parsedIndex;
			} else if (data && maxIndex === parsedIndex) {
				maxIndex = parsedIndex + 1;
			}
		});

		return `${coreName} (${maxIndex})`;
	}

	return `${coreName} (1)`;
};
