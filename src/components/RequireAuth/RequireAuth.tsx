import React from 'react';
import { useLocation, Navigate } from 'react-router-dom';
import { useSelector } from 'react-redux';
import { RootStore } from '../../redux';

export const RequireAuth: React.FC<{ children: JSX.Element }> = ({
	children,
}) => {
	const location = useLocation();
	const accessToken = useSelector(
		(state: RootStore) => state.auth.token.accessToken,
	);

	if (!accessToken) {
		// Redirect them to the /login page, but save the current location they were
		// trying to go to when they were redirected. This allows us to send them
		// along to that page after they login, which is a nicer user experience
		// than dropping them off on the home page.
		return <Navigate to="/sign-in" state={{ from: location }} replace />;
	}

	return children;
};
