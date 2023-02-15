import { Flex, Td, Text, Tr, HStack, IconButton } from '@chakra-ui/react';
import React from 'react';
import { QueueModal } from './QueueModal';
import { QueueTypeWithIndex } from '../../../atoms/atoms.types';
import { AlertDialog } from '../../AlertDialog/AlertDialog';
import { FaTrashAlt } from 'react-icons/fa';
import { useRecoilState } from 'recoil';
import * as atom from '../../../atoms/atoms';
import axios from 'axios';

export const QueueRow: React.FC<QueueTypeWithIndex> = props => {
	const [queue, setQueue] = useRecoilState(atom.queueState);

	const removeFromQueue = (index: number) => {
		axios
			.post(
				'http://127.0.0.1:5300/remove_from_queue',
				{ id: queue[index - 1].id },
				{ headers: { 'Content-Type': 'application/json' } },
			)
			.then(result => {
				if (result.data.status === 'Success') {
					setQueue(result.data.content.queue);
				}
			});
	};

	return (
		<Tr>
			<Td
				border={props.lastItem ? 'none' : undefined}
				borderBottomColor="gray.600">
				<Flex
					align="center"
					flexWrap="nowrap"
					minWidth="100%"
					py=".8rem">
					<Text
						fontSize="sm"
						fontWeight="normal"
						minWidth="100%"
						noOfLines={1}>
						{props.index}
					</Text>
				</Flex>
			</Td>

			<Td
				border={props.lastItem ? 'none' : undefined}
				borderBottomColor="gray.600"
				minWidth={{ sm: '250px' }}
				ps="0px">
				<Flex
					align="center"
					flexWrap="nowrap"
					minWidth="100%"
					py=".8rem">
					<Text
						fontSize="sm"
						fontWeight="normal"
						minWidth="100%"
						noOfLines={1}>
						{props.text_prompts}
					</Text>
				</Flex>
			</Td>

			<Td
				border={props.lastItem ? 'none' : undefined}
				borderBottomColor="gray.600"
				minWidth={{ sm: '250px' }}
				ps="0px">
				<Flex
					align="center"
					flexWrap="nowrap"
					minWidth="100%"
					py=".8rem">
					<Text
						fontSize="sm"
						fontWeight="normal"
						minWidth="100%"
						noOfLines={1}>
						{props.ckpt}
					</Text>
				</Flex>
			</Td>

			{props.width === 0 ? (
				<Td
					border={props.lastItem ? 'none' : undefined}
					borderBottomColor="gray.600">
					<Text fontSize="sm" fontWeight="bold" pb=".5rem">
						Init Image
					</Text>
				</Td>
			) : (
				<Td
					border={props.lastItem ? 'none' : undefined}
					borderBottomColor="gray.600">
					<Text fontSize="sm" fontWeight="bold" pb=".5rem">
						{props.width}x{props.height}
					</Text>
				</Td>
			)}

			<Td
				border={props.lastItem ? 'none' : undefined}
				borderBottomColor="gray.600">
				<Text fontSize="sm" fontWeight="bold" pb=".5rem">
					x{props.n_iter}
				</Text>
			</Td>

			<Td
				border={props.lastItem ? 'none' : undefined}
				borderBottomColor="gray.600">
				<HStack>
					<QueueModal {...props} />

					<AlertDialog
						triggerComponent={
							<IconButton
								aria-label="Remove from Queue"
								variant="ghost"
								icon={<FaTrashAlt />}
							/>
						}
						title="Remove From Queue"
						acceptButtonText="Remove"
						acceptCallback={() => removeFromQueue(props.index)}>
						Are you sure you want to remove this item from the
						queue?
					</AlertDialog>
				</HStack>
			</Td>
		</Tr>
	);
};
