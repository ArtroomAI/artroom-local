import { useNavigate } from 'react-router-dom';
import { History } from '../../constants';

export const NavigateSetter = () => {
	History.navigate = useNavigate();

	return null;
};
