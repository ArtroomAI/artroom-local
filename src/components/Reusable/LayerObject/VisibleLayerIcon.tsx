import React from 'react';
import { FaEye, FaEyeSlash } from 'react-icons/fa';

interface IVisibleLayerIconProps {
	isChecked: boolean;
}

export const VisibleLayerIcon: React.FC<IVisibleLayerIconProps> = ({
	isChecked,
}) => <>{isChecked ? <FaEye /> : <FaEyeSlash />}</>;
