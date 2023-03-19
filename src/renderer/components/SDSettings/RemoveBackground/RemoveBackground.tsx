import React, { useContext} from 'react';
import { useRecoilState, useRecoilValue} from 'recoil';
import * as atom from '../../../atoms/atoms';
import {
    VStack,
    Button,
    Checkbox,
    FormControl,
    FormLabel,
    HStack,
    Select
} from '@chakra-ui/react';
import {initImageState, removeBackgroundState, useRemovedBackgroundState} from '../../../SettingsManager';
import { SocketContext } from '../../../socket';
import RemoveBackgroundPreview from './RemoveBackgroundPreview';

const RemoveBackground = () => {
    const socket = useContext(SocketContext);
    const [removeBackground, setRemoveBackground] = useRecoilState(removeBackgroundState);
    const [removeBackgroundPreview, setRemoveBackgroundPreview] = useRecoilState(atom.removeBackgroundPreviewState);
    const [useRemovedBackground, setUseRemovedBackground] = useRecoilState(useRemovedBackgroundState);
    const initImage = useRecoilValue(initImageState);

    return (
        <VStack>
        <FormControl className="remove-background-input">
            <HStack>
                <FormLabel htmlFor="Remove Background">
                    Background Removal Mode
                </FormLabel>
            </HStack>
            <HStack>
                <Select
                    id="remove_background"
                    name="remove_background"
                    onChange={(event) => setRemoveBackground(event.target.value)}
                    value={removeBackground}
                    variant="outline"
                >
                    <option value="face">
                        Face
                    </option>

                    <option value="u2net">
                        Standard
                    </option>

                    <option value="u2net_human_seg">
                        Human 
                    </option>
                </Select>
                <Button
                    variant='outline'
                    disabled={initImage.length == 0}
                    onClick={()=>{
                        socket.emit('preview_remove_background', {initImage, remove_background: removeBackground})
                    }
                }
            >
                Preview
                </Button>
            </HStack>
        </FormControl>
        <HStack>
        <FormLabel htmlFor="use_random_seed">
            Remove Background in Gen
        </FormLabel>
        <Checkbox
            id="use_removed_background"
            isChecked={useRemovedBackground}
            name="use_removed_background"
            onChange={() => {
                setUseRemovedBackground((useRemovedBackground) => !useRemovedBackground);
            }}
            pb="12px"
            />

        </HStack>

        {removeBackgroundPreview.length > 0 && 
            <RemoveBackgroundPreview></RemoveBackgroundPreview> 
        }
    </VStack>
    );
};

export default RemoveBackground;
