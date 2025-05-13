# Stucture of the Dataset `meta-kaggle`

## Submissions

- SubmittedUserId --- `Users`
- TeamId --- `Teams`
- SourceKernelVersionId --- `Kernels`

## Teams

> When you accept the rules and join a Competition, you automatically do so as part of a new team consisting solely of yourself. You can then adjust your team settings in various ways by visiting the “Team” tab on the Competition page.
>
> Source: https://www.kaggle.com/docs/competitions.
>
> Observation from the data:
> **Teams are different for different Contests.**

- CompetitionId --- `Competitions`
- TeamLeaderId --- `Users`
- PublicLeaderboardSubmissionId --- ?
- PrivateLeaderboardSubmissionId --- ?
- WriteUpForumTopicId --- ?

## TeamMemberships

- TeamId --- `Teams`
- UserId --- `Users`

## Kernels

- AuthorUserId --- `Users`
- CurrentKernelVersionId --- ?
- ForkParentKernelVersionId --- ?
- ForumTopicId --- ?
- FirstKernelVersionId --- ?

## Models

- OwnerUserId --- `Users`
- OwnerOrganizationId --- `Organization`
- CurrentModelVersionId --- `ModelVersions`
- ForumId --- `Forum`

## ModelVersions

- ModelId --- `Models`
- CreatorUserId --- `Users`

## Competitions

- ForumId --- `Forums`
- OrganizationId --- `Organization`
- CompetitionTypeId --- ?

## Forums

- ParentForumId --- `Forums`

## Datasets

- CreatorUserId --- `Users`
- OwnerUserId --- `Users`
- OwnerOrganizationId --- `Organization`
- CurrentDatasetVersionId --- ?
- CurrentDatasourceVersionId --- ?
- ForumId --- `Forum`

## Tags

- ParentTagId --- `Tags`

## Episodes

- CompetitionId --- `Competition`

## EpisodeAgents

- EpisodeId --- `Episode`
- SubmissionId --- `Submission`
