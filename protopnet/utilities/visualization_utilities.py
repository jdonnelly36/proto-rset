def indices_to_upsampled_boxes(indices, latent_size, image_size, align_corners=True):
    """
    Maps indices from (-1, 1) to the size of the image
    """
    # Mapped to 0, 1
    indices = (indices + 1) / 2

    if align_corners:
        box_shape = (image_size[0] / latent_size[0], image_size[1] / latent_size[1])
        reduced_image_size = (
            image_size[0] - box_shape[0],
            image_size[1] - box_shape[1],
        )

        box_tl = (
            int(indices[0] * reduced_image_size[0]),
            int(indices[1] * reduced_image_size[1]),
        )
        box_br = (
            int(indices[0] * reduced_image_size[0] + box_shape[0]),
            int(indices[1] * reduced_image_size[1] + box_shape[1]),
        )
    else:
        box_shape = (
            image_size[0] / (latent_size[0] - 1),
            image_size[1] / (latent_size[1] - 1),
        )
        box_tl = (
            int(indices[0] * image_size[0] - box_shape[0] / 2),
            int(indices[1] * image_size[1] - box_shape[1] / 2),
        )
        box_br = (
            int(indices[0] * image_size[0] + box_shape[0] / 2),
            int(indices[1] * image_size[1] + box_shape[1] / 2),
        )

    return box_tl, box_br
