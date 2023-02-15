import { FC } from 'react';
import { FaTrash } from 'react-icons/fa';
import { Button } from '@chakra-ui/react';
import { useRecoilValue, useSetRecoilState } from 'recoil';
import { AlertDialog } from '../../AlertDialog/AlertDialog';
import {
	clearCanvasHistoryAction,
	isStagingSelector,
} from '../atoms/canvas.atoms';

export const ClearCanvasHistoryButtonModal: FC = () => {
	const clearCanvasHistory = useSetRecoilState(clearCanvasHistoryAction);
	const isStaging = useRecoilValue(isStagingSelector);

	return (
		<AlertDialog
			title="Clear Canvas History"
			acceptCallback={clearCanvasHistory}
			acceptButtonText="Clear History"
			triggerComponent={
				<Button size="sm" leftIcon={<FaTrash />} isDisabled={isStaging}>
					Clear Canvas History
				</Button>
			}>
			<p>
				Clearing the canvas history leaves your current canvas intact,
				but irreversibly clears the undo and redo history.
			</p>
			<br />
			<p>Are you sure you want to clear the canvas history?</p>
		</AlertDialog>
	);
};
