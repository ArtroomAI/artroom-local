import { FC } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaUndo } from 'react-icons/fa';
import { useSetRecoilState, useRecoilValue } from 'recoil';
import { IconButton } from '../../components';
import { undoAction, canUndoSelector } from '../../atoms/canvas.atoms';

export const CanvasUndoButton: FC = () => {
	const undo = useSetRecoilState(undoAction);
	const canUndo = useRecoilValue(canUndoSelector);

	const handleUndo = () => {
		undo();
	};

	useHotkeys(
		['meta+z', 'ctrl+z'],
		() => {
			handleUndo();
		},
		{
			enabled: () => canUndo,
			preventDefault: true,
		},
		[canUndo],
	);

	return (
		<IconButton
			aria-label="Undo (Ctrl+Z)"
			tooltip="Undo (Ctrl+Z)"
			icon={<FaUndo />}
			onClick={handleUndo}
			isDisabled={!canUndo}
		/>
	);
};
