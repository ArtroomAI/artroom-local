export const getPictureUrl = (hash?: string | null) => {
	return hash
		? `${
				import.meta.env.VITE_BASE_URL
		  }api/pictures/item-hash?itemHash=${hash}`
		: undefined;
};
