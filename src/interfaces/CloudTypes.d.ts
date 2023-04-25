enum CloudStatus {
    'PENDING' = 'PENDING',
    'SUCCESS' = 'SUCCESS',
    'FAILED' = 'FAILED'
}
interface CloudImage {
    id: number;
    url: string;
    status: CloudStatus;
}
interface CloudJob {
    id: number;
    images: CloudImage[]
    image_settings: {
        shard_cost: number;
        n_iter: number;
    }
}
interface CloudGetStatus {
    data: {
        jobs: CloudJob[];
        shards: number;
    }
}
