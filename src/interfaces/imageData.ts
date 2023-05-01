import type { IValidation } from "typia";

export type ExifValidation = IValidation<ExifDataType>;

export interface BackEndImage {
    b64: string;
    metadata: ExifValidation | null;
}
