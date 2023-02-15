import ReactGA from 'react-ga4';

export const onSendAnalyticEvent = (
	label: string | undefined,
	action: string,
	category: string,
) => {
	ReactGA.event({
		action,
		category,
		label,
	});
};
