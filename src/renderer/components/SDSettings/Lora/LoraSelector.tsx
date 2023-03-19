import React from "react";
import {
  Select,
  VStack,
  HStack,
  Input,
  IconButton,
  Icon,
  Text,
  Flex,
  FormControl,
  FormLabel
} from "@chakra-ui/react";
import { FaTimes } from "react-icons/fa";
import { IoMdCloud } from "react-icons/io";
import { useRecoilState } from "recoil";
import { loraState } from "../../../SettingsManager";

function LoraSelector({options, cloudMode}: {options: any[], cloudMode: boolean}) {
    const [lora, setLora] = useRecoilState(loraState);

    const handleAddItem = (item: {name: string, weight: number}) => {
        if (!lora.some((i) => i.name === item.name)) {
            setLora([...lora, { ...item, weight: 1 }]);
        }
    };

    const handleRemoveItem = (itemToRemove: {name: string, weight: number}) => {
        setLora(
            lora.filter((item) => item.name !== itemToRemove.name)
        );
    };

    const handleweightChange = (itemToUpdate: {name: string, weight: number}, weight: number) => {
        const newItems = lora.map((item) => {
            if (item.name === itemToUpdate.name) {
            return { ...item, weight };
            }
            return item;
        });
        setLora(newItems);
    };

  return (
    <>
    <FormControl className="lora-input">
        <FormLabel htmlFor="Lora">
            <HStack>
                <Text>
                    Choose Your Lora
                </Text>
                {cloudMode
                    ? <Icon as={IoMdCloud} />
                    : null}
            </HStack>
        </FormLabel>
      <Select
            placeholder="Select item"
            onChange={(event) =>{
                if (event.target.value.length){
                    handleAddItem({ name: event.target.value, weight: 1 })
                }
            }
        }
      >
        {options.map((item) => (
          <option key={item} value={item}>
            {item}
          </option>
        ))}
      </Select>

      <VStack pt={4}>
        {lora.length &&
            <HStack>
                <Text fontWeight="normal" fontSize="sm" width="120px"> 
                    Name
                </Text>
                <Text fontWeight="normal" fontSize="sm" width="70px">
                    Weight
                </Text>
            </HStack>
        }
        {lora.map((item) => (
            <Flex align={'center'} key={item.name}>
                <Text fontWeight="normal" width="120px" pr="15px">
                    {item.name}
                </Text>
                <Input
                    width="70px"
                    type="number"
                    min={0}
                    value={item.weight}
                    onChange={(event) =>
                    handleweightChange(item, parseFloat(event.target.value))
                    }
                />
                <IconButton
                    aria-label="Remove"
                    icon={<Icon as={FaTimes} />}
                    variant="ghost"
                    size='sm'
                    onClick={() => handleRemoveItem(item)}
                />
            </Flex>
        ))}
      </VStack>
    </FormControl>
    </>
  );
}

export default LoraSelector;