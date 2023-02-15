import React, { useEffect, useRef, useState } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../../atoms/atoms';
import {
	Box,
	Image,
	IconButton,
	ButtonGroup,
	useColorModeValue,
	Text,
} from '@chakra-ui/react';
import { FiUpload } from 'react-icons/fi';
import { FaTrashAlt } from 'react-icons/fa';

export const DragDropFile: React.FC = () => {
	const [dragActive, setDragActive] = useState(false);
	const inputRef = useRef(null);
	const [imageSettings, setImageSettings] = useRecoilState(
		atom.imageSettingsState,
	);
	const [initImagePath, setInitImagePath] = useRecoilState(
		atom.initImagePathState,
	);

	// function getImageFromPath() {
	// 	console.log(initImagePath, 'initImagePath');

	// 	if (initImagePath.length > 0) {
	// 		// alert('window.api.getImageFromPath placeholder');
	// 		console.log(initImagePath);
	// 		// window.api.getImageFromPath(initImagePath).then(result => {
	// 		// 	setImageSettings({
	// 		// 		...imageSettings,
	// 		// 		init_image: result.b64,
	// 		// 	});
	// 		// });
	// 	}
	// }

	// useEffect(() => {
	// 	getImageFromPath();
	// }, [initImagePath]);

	// Handle drag events
	const handleDrag: React.DragEventHandler<HTMLElement> = function (e) {
		e.preventDefault();
		e.stopPropagation();
		if (e.type === 'dragenter' || e.type === 'dragover') {
			setDragActive(true);
		} else if (e.type === 'dragleave') {
			setDragActive(false);
		}
	};

	// Triggers when file is dropped
	const handleDrop: React.DragEventHandler<HTMLDivElement> = function (e) {
		e.preventDefault();
		e.stopPropagation();
		setDragActive(false);
		if (e.dataTransfer.files && e.dataTransfer.files[0]) {
			handleFile(e.dataTransfer.files);
		}
	};

	const handleFile = function (e: FileList) {
		console.log(e[0], URL.createObjectURL(e[0]), 'e[0]');

		setImageSettings({
			...imageSettings,
			init_image: URL.createObjectURL(e[0]),
		});
		// ORIGINAL
		// setInitImagePath(e[0].path);

		// setInitImagePath(e[0].name);
	};

	// Triggers when file is selected with click
	const handleChange: React.ChangeEventHandler<HTMLInputElement> = e => {
		e.preventDefault();
		if (e.target.files && e.target.files[0]) {
			handleFile(e.target.files);
		}
	};

	// Triggers the input when the button is clicked
	const onButtonClick = () => {
		// @ts-ignore
		inputRef.current?.click();
	};

	return (
		<Box
			bg={useColorModeValue('white', 'background')}
			height="140px"
			width="140px">
			<Box
				border="1px"
				borderStyle="ridge"
				height="140px"
				onClick={onButtonClick}
				onDragEnter={handleDrag}
				onDragLeave={handleDrag}
				onDragOver={handleDrag}
				onDrop={handleDrop}
				rounded="md"
				style={{
					display: 'flex',
					alignItems: 'center',
					justifyContent: 'center',
					textAlign: 'center',
					borderColor: useColorModeValue('#e2e8f0', '#FFFFFF20'),
				}}
				width="140px">
				{imageSettings.init_image.length > 0 ? (
					<Image
						boxSize="140px"
						fit="contain"
						rounded="md"
						src={imageSettings.init_image}
					/>
				) : (
					<Text fontSize="xs">Click or Drag Image Here</Text>
				)}
			</Box>

			<form
				id="form-file-upload"
				onDragEnter={handleDrag}
				onSubmit={e => e.preventDefault()}>
				<input
					accept="image/png, image/jpeg"
					id="input-file-upload"
					className="input-file-upload"
					multiple={false}
					onChange={handleChange}
					ref={inputRef}
					type="file"
				/>

				<label htmlFor="input-file-upload" id="label-file-upload">
					<ButtonGroup isAttached variant="outline" my="20px">
						<IconButton
							// border="2px"
							icon={<FiUpload />}
							onClick={onButtonClick}
							width="100px"
							aria-label="upload"
						/>

						<IconButton
							aria-label="Clear Init Image"
							// border="2px"
							icon={<FaTrashAlt />}
							onClick={() => {
								setInitImagePath('');
								setImageSettings({
									...imageSettings,
									init_image: '',
								});
							}}
						/>
					</ButtonGroup>
				</label>
			</form>
		</Box>
	);
};
