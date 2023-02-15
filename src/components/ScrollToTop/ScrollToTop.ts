import { useLayoutEffect } from 'react';
import { useLocation } from 'react-router-dom';

export const ScrollToTop: React.FC<{ children: JSX.Element }> = ({
	children,
}) => {
	const location = useLocation();
	useLayoutEffect(() => {
		if (!location.pathname.includes('/img/')) {
			document.documentElement.scrollTo(0, 0);
		}
	}, [location.pathname]);
	return children;
};
